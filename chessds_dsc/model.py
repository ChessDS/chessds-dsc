# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""The DSC network: a square-token transformer with a from x to attention policy head.

The architecture is Chessformer (Monroe, Eilender, Chalmers, Tang and Anderson, "Chessformer: A
Unified Architecture for Chess Modeling", ICLR 2026, arXiv 2605.19091), implemented independently
from the paper's description, with four additions: embeddings for the time control, the rating
system and the venue, and an embedding of how many previous boards are present.

* Input: 64 square tokens of ``TOKEN_DIM`` channels (:mod:`.encoding`), projected together with a
  per-position conditioning vector. The conditioning concatenates two "soft Elo" embeddings
  (``w + (s - w) * clamp(elo / 5000, 0, 1)`` with learned endpoints ``w`` and ``s``, one pair for the
  mover and one for the opponent) and the sum of the four conditioning embeddings. There is no
  learned square embedding by default: positional information comes from the attention bias.
* ``n_layers`` pre-LayerNorm encoder blocks: multi-head self-attention with a dynamic geometric
  attention bias (a small network reads all tokens and emits a 64 x 64 logit offset per head,
  through a projection shared by every layer), then a GELU MLP.
* Policy head: query (from-square) and key (to-square) projections give a 64 x 64 logit table; a
  linear read-out of the last-rank key vectors adds rook, bishop and knight promotion offsets.
  Logits are flattened in :func:`.encoding.move_index` order with ``-inf`` on illegal moves.
* Auxiliary WDL head: mean over tokens -> LayerNorm -> Linear -> ReLU -> Linear(3), giving
  mover-relative [loss, draw, win] logits.

Parameter names match the published checkpoints; see :mod:`.checkpoint` for the file format.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoding import (
    N_BOARDS,
    N_MOVES,
    N_RATING_SYSTEMS,
    N_SQUARES,
    N_TIME_CONTROLS,
    N_VENUES,
    TOKEN_DIM,
)

ELO_SCALE = 5000.0
N_PROMO_CHANNELS = 3  # rook, bishop, knight offsets on top of the none / queen logit
N_WDL = 3
_LAST_RANK_START = N_SQUARES - 8  # to-squares 56..63 in mover orientation
_PENULTIMATE_RANK_START = N_SQUARES - 16  # from-squares 48..55


@dataclass(frozen=True)
class DSCConfig:
    """Architecture hyper-parameters. ``preset("23m")`` etc. give the published sizes."""

    width: int
    n_layers: int
    head_dim: int = 32
    mlp_mult: int = 2
    elo_dim: int = 128
    cond_dim: int = 64
    gab_compress: int = 32  # per-token compression of the attention-bias generator
    gab_hidden: int = 128  # its hidden width
    gab_head_dim: int = 128  # bias templates per head
    policy_dim: int | None = None  # None = encoder width
    wdl_hidden: int = 128
    square_embedding: bool = False  # learned absolute square embedding (off in every DSC model)

    def __post_init__(self) -> None:
        if self.width % self.head_dim != 0:
            raise ValueError(f"width {self.width} is not a multiple of head_dim {self.head_dim}")

    @property
    def n_heads(self) -> int:
        return self.width // self.head_dim

    @property
    def policy_width(self) -> int:
        return self.width if self.policy_dim is None else self.policy_dim

    @classmethod
    def preset(cls, name: str) -> DSCConfig:
        presets = {
            "23m": cls(width=512, n_layers=8),
            "79m": cls(width=1024, n_layers=8),
            "150m": cls(width=1024, n_layers=16),
        }
        if name not in presets:
            raise KeyError(f"unknown preset {name!r}; choose from {sorted(presets)}")
        return presets[name]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DSCConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown architecture fields: {sorted(unknown)}")
        return cls(**d)


class SoftElo(nn.Module):
    """``w + (s - w) * clamp(elo / 5000, 0, 1)`` with learned endpoints ``w`` and ``s``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))
        self.s = nn.Parameter(torch.zeros(dim))

    def forward(self, elo: Tensor) -> Tensor:
        t = torch.clamp(elo.float() / ELO_SCALE, 0.0, 1.0).unsqueeze(-1)
        return self.w + (self.s - self.w) * t


class GeometricBias(nn.Module):
    """Input-dependent attention bias: one ``(64, 64)`` logit offset per head and position."""

    def __init__(self, cfg: DSCConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.gab_head_dim
        self.compress = nn.Linear(cfg.width, cfg.gab_compress, bias=False)
        self.hidden = nn.Linear(N_SQUARES * cfg.gab_compress, cfg.gab_hidden)
        self.ln_hidden = nn.LayerNorm(cfg.gab_hidden)
        self.heads = nn.Linear(cfg.gab_hidden, cfg.n_heads * cfg.gab_head_dim)
        self.ln_heads = nn.LayerNorm(cfg.n_heads * cfg.gab_head_dim)

    def forward(self, x: Tensor, shared: Tensor) -> Tensor:
        """``x (B, 64, width)``, ``shared (gab_head_dim, 4096)`` -> ``(B, heads, 64, 64)``."""
        b = x.shape[0]
        z = self.compress(x).reshape(b, -1)
        z = self.ln_hidden(F.gelu(self.hidden(z)))
        z = self.ln_heads(F.gelu(self.heads(z)))
        z = z.reshape(b, self.n_heads, self.head_dim)
        return torch.matmul(z, shared.to(z.dtype)).reshape(b, self.n_heads, N_SQUARES, N_SQUARES)


class Block(nn.Module):
    """Pre-LayerNorm encoder block: biased self-attention, then a GELU MLP."""

    def __init__(self, cfg: DSCConfig) -> None:
        super().__init__()
        d = cfg.width
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.gab = GeometricBias(cfg)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, cfg.mlp_mult * d)
        self.fc2 = nn.Linear(cfg.mlp_mult * d, d)

    def forward(self, x: Tensor, shared: Tensor) -> Tensor:
        b, n, d = x.shape
        h = self.ln1(x)
        bias = self.gab(h, shared)
        qkv = self.qkv(h).reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        a = a.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class PolicyHead(nn.Module):
    """from x to attention logits, with promotion offsets read from the last-rank key vectors."""

    def __init__(self, cfg: DSCConfig) -> None:
        super().__init__()
        pw = cfg.policy_width
        self.scale = 1.0 / math.sqrt(pw)
        self.q = nn.Linear(cfg.width, pw)
        self.k = nn.Linear(cfg.width, pw)
        self.promo = nn.Linear(pw, N_PROMO_CHANNELS)
        squares = torch.arange(N_SQUARES)
        last_rank = (squares >= _LAST_RANK_START).float().reshape(1, N_SQUARES, 1)
        penultimate = (squares >= _PENULTIMATE_RANK_START) & (squares < _LAST_RANK_START)
        promo_pairs = penultimate.float().reshape(N_SQUARES, 1) * last_rank.reshape(1, N_SQUARES)
        self.register_buffer("last_rank", last_rank, persistent=False)
        self.register_buffer(
            "promo_pairs", promo_pairs.reshape(1, N_SQUARES, N_SQUARES, 1), persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """``x (B, 64, width)`` -> ``(B, N_MOVES)`` unmasked logits in move-index order."""
        b = x.shape[0]
        q, k = self.q(x), self.k(x)
        base = torch.matmul(q, k.transpose(1, 2)) * self.scale  # (B, from, to)
        promo = self.promo(k) * self.last_rank.to(k.dtype)  # (B, to, 3), zero off the last rank
        offset = promo.unsqueeze(1) * self.promo_pairs.to(k.dtype)  # (B, from, to, 3)
        base = base.unsqueeze(-1)
        logits = torch.cat([base, base + offset], dim=-1)  # (B, from, to, 4)
        return logits.reshape(b, N_MOVES)


class WdlHead(nn.Module):
    def __init__(self, cfg: DSCConfig) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(cfg.width)
        self.fc1 = nn.Linear(cfg.width, cfg.wdl_hidden)
        self.fc2 = nn.Linear(cfg.wdl_hidden, N_WDL)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(self.ln(x.mean(dim=1)))))


class DSCModel(nn.Module):
    def __init__(self, cfg: DSCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.elo_mover = SoftElo(cfg.elo_dim)
        self.elo_opp = SoftElo(cfg.elo_dim)
        self.emb_tc = nn.Embedding(N_TIME_CONTROLS, cfg.cond_dim)
        self.emb_rsys = nn.Embedding(N_RATING_SYSTEMS, cfg.cond_dim)
        self.emb_venue = nn.Embedding(N_VENUES, cfg.cond_dim)
        self.emb_hist = nn.Embedding(N_BOARDS, cfg.cond_dim)  # number of real previous boards
        self.in_proj = nn.Linear(TOKEN_DIM + 2 * cfg.elo_dim + cfg.cond_dim, d)
        if cfg.square_embedding:
            self.square_emb = nn.Parameter(torch.zeros(N_SQUARES, d))
        else:
            self.register_buffer("square_emb", torch.zeros(1, d), persistent=False)
        self.gab_shared = nn.Parameter(torch.zeros(cfg.gab_head_dim, N_SQUARES * N_SQUARES))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.policy = PolicyHead(cfg)
        self.wdl = WdlHead(cfg)
        token_scale = torch.ones(TOKEN_DIM)
        token_scale[TOKEN_DIM - 1] = 0.01  # the halfmove clock is stored as min(clock, 100)
        self.register_buffer("token_scale", token_scale, persistent=False)

    def conditioning(
        self,
        elo_mover: Tensor,
        elo_opp: Tensor,
        tc: Tensor,
        rsys: Tensor,
        venue: Tensor,
        hist_count: Tensor,
    ) -> Tensor:
        cond = (
            self.emb_tc(tc.long())
            + self.emb_rsys(rsys.long())
            + self.emb_venue(venue.long())
            + self.emb_hist(hist_count.long())
        )
        return torch.cat([self.elo_mover(elo_mover), self.elo_opp(elo_opp), cond], dim=-1)

    def embed(self, tokens: Tensor, cond: Tensor) -> Tensor:
        """``tokens uint8 (B, 64, TOKEN_DIM)``, ``cond (B, 2 * elo_dim + cond_dim)`` ->
        ``(B, 64, width)``: one linear layer over the token channels and the conditioning
        broadcast to every square, computed as two matrix products."""
        feats = tokens.to(torch.float32) * self.token_scale
        w_tok = self.in_proj.weight[:, :TOKEN_DIM]
        w_cond = self.in_proj.weight[:, TOKEN_DIM:]
        x = F.linear(feats, w_tok, self.in_proj.bias) + F.linear(cond, w_cond).unsqueeze(1)
        return x + self.square_emb

    def forward(
        self,
        tokens: Tensor,
        elo_mover: Tensor,
        elo_opp: Tensor,
        tc: Tensor,
        rsys: Tensor,
        venue: Tensor,
        hist_count: Tensor,
        legal: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """-> ``(policy_logits float32 (B, N_MOVES) with -inf on illegal moves,
        wdl_logits float32 (B, 3) mover-relative [loss, draw, win])``."""
        cond = self.conditioning(elo_mover, elo_opp, tc, rsys, venue, hist_count)
        x = self.embed(tokens, cond)
        for block in self.blocks:
            x = block(x, self.gab_shared)
        x = self.ln_f(x)
        policy_logits = self.policy(x).to(torch.float32)
        policy_logits = policy_logits.masked_fill(~legal.to(torch.bool), float("-inf"))
        wdl_logits = self.wdl(x).to(torch.float32)
        return policy_logits, wdl_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
