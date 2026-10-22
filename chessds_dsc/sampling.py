# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Choosing a move from a probability distribution (temperature and nucleus sampling)."""

from __future__ import annotations

import numpy as np


def select_move(
    moves: dict[str, float],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> str:
    """Pick a move from ``moves`` (uci -> probability).

    ``temperature <= 0`` returns the most probable move. Otherwise the probabilities are
    sharpened or flattened as ``p ** (1 / temperature)`` and renormalised; with ``top_p < 1`` only
    the most probable moves whose cumulative probability stays within ``top_p`` are kept (the top
    move always is), then one is drawn.
    """
    if not moves:
        raise ValueError("no moves to choose from")
    ucis = list(moves)
    probs = np.asarray([moves[u] for u in ucis], dtype=np.float64)
    if temperature <= 0:
        return ucis[int(np.argmax(probs))]
    rng = rng or np.random.default_rng()
    with np.errstate(divide="ignore"):
        logp = np.log(np.clip(probs, 0.0, None)) / temperature
    logp -= logp.max()
    scaled = np.exp(logp)
    scaled /= scaled.sum()
    order = np.argsort(-scaled, kind="stable")
    if top_p < 1.0:
        sorted_p = scaled[order]
        keep = np.cumsum(sorted_p) <= top_p
        keep[0] = True
        order = order[keep]
    kept = scaled[order]
    kept /= kept.sum()
    choice = rng.choice(len(order), p=kept)
    return ucis[int(order[choice])]
