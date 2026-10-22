from __future__ import annotations

import chess
import numpy as np
import pytest
import torch
from chessds_dsc import encoding as E
from chessds_dsc.model import DSCConfig, DSCModel, count_parameters
from conftest import TINY, boards_after, tiny_model


def inputs(boards_list, elo=(2578.0, 2578.0)):
    tokens = torch.from_numpy(np.stack([E.encode_position(b) for b in boards_list]))
    legal = torch.from_numpy(np.stack([E.legal_mask(b[0]) for b in boards_list]))
    n = len(boards_list)
    return dict(
        tokens=tokens,
        legal=legal,
        elo_mover=torch.full((n,), elo[0]),
        elo_opp=torch.full((n,), elo[1]),
        tc=torch.full((n,), E.TIME_CONTROL_CODES["classical"], dtype=torch.int64),
        rsys=torch.full((n,), E.RATING_SYSTEM_CODES["fide"], dtype=torch.int64),
        venue=torch.full((n,), E.VENUE_CODES["otb"], dtype=torch.int64),
        hist_count=torch.tensor([E.hist_count_of(b) for b in boards_list]),
    )


def test_shapes_masking_and_determinism():
    model = tiny_model()
    batch = [boards_after("e4 e5 Nf3"), boards_after("d4 d5")]
    kw = inputs(batch)
    with torch.no_grad():
        policy, wdl = model(**kw)
        policy2, wdl2 = model(**kw)
    assert policy.shape == (2, E.N_MOVES) and wdl.shape == (2, 3)
    assert policy.dtype == torch.float32 and wdl.dtype == torch.float32
    assert torch.equal(policy, policy2) and torch.equal(wdl, wdl2)
    probs = torch.softmax(policy, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5)
    assert (probs[~kw["legal"]] == 0).all()
    assert torch.isfinite(policy[kw["legal"]]).all()


def test_promotion_channels_only_affect_last_rank_moves():
    model = tiny_model()
    board = chess.Board("r3k2r/1P6/8/8/8/8/8/R3K2R w KQkq - 0 1")
    kw = inputs([[board]])
    kw["legal"] = torch.ones(1, E.N_MOVES, dtype=torch.bool)  # inspect every logit
    with torch.no_grad():
        policy, _ = model(**kw)
    logits = policy.reshape(64, 64, 4)
    # Non-promotion from/to pairs: the four channels are identical.
    assert torch.allclose(logits[:48, :, 0:1].expand(-1, -1, 4), logits[:48, :, :])
    assert torch.allclose(logits[:, :56, 0:1].expand(-1, -1, 4), logits[:, :56, :])
    # Promotion pairs (from 48..55 to 56..63) differ across channels.
    assert not torch.allclose(logits[48:56, 56:64, 0], logits[48:56, 56:64, 1])


def test_ratings_and_history_change_the_output_in_eval():
    model = tiny_model()
    a = boards_after("e4 e5 Nf3 Nf6 Nc3 Nc6")
    kw = inputs([a])
    with torch.no_grad():
        p0, _ = model(**kw)
        p1, _ = model(**{**kw, "elo_mover": torch.tensor([1500.0])})
        p2, _ = model(**inputs([[a[0]]]))
    legal = kw["legal"][0]
    assert not torch.allclose(p0[0][legal], p1[0][legal])
    assert not torch.allclose(p0[0][legal], p2[0][legal])


def test_presets_and_parameter_count():
    with torch.device("meta"):
        assert count_parameters(DSCModel(DSCConfig.preset("23m"))) == 22_538_566
        assert count_parameters(DSCModel(DSCConfig.preset("79m"))) == 77_053_254
        assert count_parameters(DSCModel(DSCConfig.preset("150m"))) == 150_907_206
    with pytest.raises(KeyError):
        DSCConfig.preset("500m")
    assert DSCConfig.from_dict(TINY.to_dict()) == TINY
    with pytest.raises(ValueError):
        DSCConfig.from_dict({**TINY.to_dict(), "bogus": 1})
    with pytest.raises(ValueError):
        DSCConfig(width=100, n_layers=1)
