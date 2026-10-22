from __future__ import annotations

import collections

import numpy as np
import pytest
from chessds_dsc.sampling import select_move

MOVES = {"e2e4": 0.5, "d2d4": 0.3, "g1f3": 0.15, "c2c4": 0.05}


def test_argmax_and_errors():
    assert select_move(MOVES, temperature=0) == "e2e4"
    assert select_move(MOVES, temperature=-1) == "e2e4"
    with pytest.raises(ValueError):
        select_move({}, temperature=0)


def test_seeded_sampling_is_reproducible_and_follows_the_distribution():
    a = [select_move(MOVES, rng=np.random.default_rng(7)) for _ in range(20)]
    b = [select_move(MOVES, rng=np.random.default_rng(7)) for _ in range(20)]
    assert a == b
    rng = np.random.default_rng(0)
    counts = collections.Counter(select_move(MOVES, rng=rng) for _ in range(20000))
    assert abs(counts["e2e4"] / 20000 - 0.5) < 0.02 and abs(counts["c2c4"] / 20000 - 0.05) < 0.01


def test_temperature_and_top_p():
    rng = np.random.default_rng(0)
    cold = collections.Counter(select_move(MOVES, temperature=0.3, rng=rng) for _ in range(5000))
    assert cold["e2e4"] / 5000 > 0.78
    hot = collections.Counter(select_move(MOVES, temperature=5.0, rng=rng) for _ in range(5000))
    assert hot["c2c4"] / 5000 > 0.15
    nucleus = {select_move(MOVES, top_p=0.6, rng=rng) for _ in range(2000)}
    assert nucleus == {"e2e4"}  # 0.5 <= 0.6 < 0.8: only the top move survives
    nucleus = {select_move(MOVES, top_p=0.85, rng=rng) for _ in range(2000)}
    assert nucleus == {"e2e4", "d2d4"}
    assert select_move(MOVES, top_p=0.01, rng=rng) == "e2e4"  # the top move is always kept
