# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Python inference API: :class:`DSC`.

    model = DSC.load("dsc-23m")
    pred = model.predict(board, white_elo=2700, black_elo=2650)
    pred.moves   # {uci: probability} over the legal moves, most probable first
    pred.wdl     # WDL(win, draw, loss) of the side to move, from the auxiliary neural head
    pred.inputs  # what the model was actually given (ratings, history, repetition, ...)

Inputs and defaults
-------------------
* ``board``: a ``chess.Board`` or a FEN string. A board that carries its move stack provides the
  previous positions (used by history models) and, when the stack covers the whole game from the
  standard start position, the exact repetition state.
* ``white_elo`` / ``black_elo``: the players' ratings. When omitted both default to
  ``DEFAULT_ELO`` = 2578, a strong-grandmaster reference level on the FIDE scale; the prediction
  is then for players of that strength, not an average over unknown strengths.
* ``time_control`` (default ``"classical"``), ``rating_system`` (default ``"fide"``; the scale the
  ratings are on) and ``venue`` (derived from the rating system unless given): the game context
  the model was trained to condition on.
* ``use_history``: for history models, ``None`` (default) uses the previous positions the board
  carries, ``True`` requires them and ``False`` ignores them. Strict models never use history.
* ``repetition``: ``(seen once before, seen twice before)`` for the current position. ``None``
  (default) reads it from a complete move stack and otherwise assumes no repetition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import chess
import numpy as np
import torch

from . import encoding as enc
from .checkpoint import load_model
from .constants import DEFAULT_ELO as DEFAULT_ELO
from .constants import DEFAULT_RATING_SYSTEM as DEFAULT_RATING_SYSTEM
from .constants import DEFAULT_TIME_CONTROL as DEFAULT_TIME_CONTROL
from .model import DSCModel
from .registry import fetch, resolve


class WDL(NamedTuple):
    """Win, draw and loss probabilities for the side to move (auxiliary neural head)."""

    win: float
    draw: float
    loss: float


@dataclass
class PositionInput:
    board: chess.Board | str
    white_elo: float | None = None
    black_elo: float | None = None
    time_control: str | None = None  # None = the model's default (classical)
    rating_system: str | None = None  # None = the model's default (fide)
    venue: str | None = None  # None = derived from the rating system
    use_history: bool | None = None
    repetition: tuple[bool, bool] | None = None


@dataclass(frozen=True)
class ResolvedInputs:
    """What the model was given for one position."""

    fen: str
    elo_mover: float
    elo_opponent: float
    time_control: str
    rating_system: str
    venue: str
    history_boards: int  # previous positions actually encoded (0..7)
    repetition: tuple[bool, bool]
    repetition_source: str  # "move_stack", "explicit" or "assumed_none"


@dataclass(frozen=True)
class Prediction:
    moves: dict[str, float]  # uci -> probability over the legal moves, most probable first
    wdl: WDL
    inputs: ResolvedInputs

    @property
    def best(self) -> str | None:
        return next(iter(self.moves), None)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DSC:
    """A loaded DSC model."""

    def __init__(self, model: DSCModel, config: dict, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        # The input defaults come from the model's own configuration, never from this package.
        defaults = config.get("defaults") or {}
        missing = [k for k in ("elo", "time_control", "rating_system") if k not in defaults]
        if missing:
            raise ValueError(f"model configuration lacks input defaults {missing}")
        self.default_elo = float(defaults["elo"])
        self.default_time_control = str(defaults["time_control"])
        self.default_rating_system = str(defaults["rating_system"])

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        name: str = "dsc-23m",
        *,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        base_url: str | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> DSC:
        """Load a registered model by id (downloading it if needed) or, with ``checkpoint_path``,
        a local ``.safetensors`` file in the DSC format (its ``.json`` must sit next to it)."""
        device = torch.device(device) if device is not None else default_device()
        if checkpoint_path is not None:
            weights = Path(checkpoint_path)
            if weights.is_dir():
                candidates = sorted(weights.glob("*.safetensors"))
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        f"{weights} must contain exactly one .safetensors file, found "
                        f"{len(candidates)}"
                    )
                weights = candidates[0]
            model, config = load_model(weights, device=device)
        else:
            entry = resolve(name)
            weights, config_path = fetch(
                entry,
                cache_dir_=cache_dir,
                base_url_=base_url,
                local_files_only=local_files_only,
                force_download=force_download,
            )
            model, config = load_model(weights, config_path, device=device)
        return cls(model, config, device)

    @property
    def model_id(self) -> str:
        return self.config.get("model_id", "dsc")

    @property
    def mode(self) -> str:
        return self.config["mode"]

    @property
    def uses_history(self) -> bool:
        return self.mode == "history"

    # ── inference ───────────────────────────────────────────────────────────

    def predict(self, board: chess.Board | str, **kwargs) -> Prediction:
        """Predict one position; keyword arguments are the :class:`PositionInput` fields."""
        return self.predict_batch([PositionInput(board, **kwargs)])[0]

    @torch.no_grad()
    def predict_batch(self, items: Sequence[PositionInput]) -> list[Prediction]:
        if not items:
            return []
        encoded = [self._encode(item) for item in items]
        n = len(encoded)
        tokens = torch.from_numpy(np.stack([e["tokens"] for e in encoded]))
        legal = torch.from_numpy(np.stack([e["legal"] for e in encoded]))
        kw = {
            "elo_mover": torch.tensor([e["elo_mover"] for e in encoded], dtype=torch.float32),
            "elo_opp": torch.tensor([e["elo_opp"] for e in encoded], dtype=torch.float32),
            "tc": torch.tensor([e["tc"] for e in encoded], dtype=torch.int64),
            "rsys": torch.tensor([e["rsys"] for e in encoded], dtype=torch.int64),
            "venue": torch.tensor([e["venue"] for e in encoded], dtype=torch.int64),
            "hist_count": torch.tensor([e["hist_count"] for e in encoded], dtype=torch.int64),
        }
        dev = self.device
        policy, wdl_logits = self.model(
            tokens.to(dev), legal=legal.to(dev), **{k: v.to(dev) for k, v in kw.items()}
        )
        log_probs = torch.log_softmax(policy.float(), dim=-1).cpu().numpy()
        wdl_probs = torch.softmax(wdl_logits.float(), dim=-1).cpu().numpy()  # [loss, draw, win]
        out = []
        for i in range(n):
            e = encoded[i]
            if e["legal_pairs"]:
                probs = {uci: float(np.exp(log_probs[i, idx])) for uci, idx in e["legal_pairs"]}
                moves = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
            else:
                moves = {}
            loss, draw, win = (float(x) for x in wdl_probs[i])
            out.append(Prediction(moves=moves, wdl=WDL(win, draw, loss), inputs=e["inputs"]))
        return out

    # ── encoding ────────────────────────────────────────────────────────────

    def _encode(self, item: PositionInput) -> dict:
        board = chess.Board(item.board) if isinstance(item.board, str) else item.board
        white = board.turn == chess.WHITE

        time_control = item.time_control or self.default_time_control
        rating_system = item.rating_system or self.default_rating_system
        if time_control not in enc.TIME_CONTROLS:
            raise ValueError(
                f"time_control must be one of {enc.TIME_CONTROLS}, got {time_control!r}"
            )
        if rating_system not in enc.RATING_SYSTEMS:
            raise ValueError(
                f"rating_system must be one of {enc.RATING_SYSTEMS}, got {rating_system!r}"
            )
        venue = item.venue or enc.VENUE_FOR_RATING_SYSTEM[rating_system]
        if venue not in enc.VENUES:
            raise ValueError(f"venue must be one of {enc.VENUES}, got {venue!r}")

        white_elo = self.default_elo if item.white_elo is None else float(item.white_elo)
        black_elo = self.default_elo if item.black_elo is None else float(item.black_elo)
        for elo in (white_elo, black_elo):
            if not np.isfinite(elo) or elo < 0:
                raise ValueError(f"ratings must be non-negative numbers, got {elo}")
        elo_mover, elo_opp = (white_elo, black_elo) if white else (black_elo, white_elo)

        if item.repetition is not None:
            repetition = (bool(item.repetition[0]), bool(item.repetition[1]))
            rep_source = "explicit"
        elif enc.rule_state_known(board):
            repetition = enc.repetition_flags(board)
            rep_source = "move_stack"
        else:
            repetition = (False, False)
            rep_source = "assumed_none"

        use_history = item.use_history
        if self.uses_history and use_history is not False:
            boards = enc.history_boards(board)
            if use_history is True and len(boards) == 1:
                raise ValueError(
                    "use_history=True but the board carries no previous positions; pass a board "
                    "with its move stack or use_history=None"
                )
        else:
            boards = [board]
        tokens = enc.encode_position(boards, repetition=repetition)
        hist_count = enc.hist_count_of(boards)

        inputs = ResolvedInputs(
            fen=board.fen(),
            elo_mover=elo_mover,
            elo_opponent=elo_opp,
            time_control=time_control,
            rating_system=rating_system,
            venue=venue,
            history_boards=hist_count,
            repetition=repetition,
            repetition_source=rep_source,
        )
        return {
            "tokens": tokens,
            "legal": enc.legal_mask(board),
            "legal_pairs": enc.legal_indices(board),
            "hist_count": hist_count,
            "elo_mover": elo_mover,
            "elo_opp": elo_opp,
            "tc": enc.TIME_CONTROL_CODES[time_control],
            "rsys": enc.RATING_SYSTEM_CODES[rating_system],
            "venue": enc.VENUE_CODES[venue],
            "inputs": inputs,
        }
