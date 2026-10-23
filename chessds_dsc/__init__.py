# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""ChessDS DSC: human chess move prediction models.

Quick start::

    import chess
    from chessds_dsc import DSC

    model = DSC.load("dsc-23m")              # downloads and caches the weights
    board = chess.Board()
    board.push_san("e4"); board.push_san("c5")
    pred = model.predict(board, white_elo=2650, black_elo=2600)
    print(pred.moves)                           # {"g1f3": 0.62, "b1c3": 0.14, ...}

The command-line UCI engine is ``dsc-uci``; see the README.
"""

from __future__ import annotations

import importlib

from .constants import DEFAULT_ELO
from .registry import MODELS, ModelEntry, resolve

__version__ = "0.1.0"

# torch is imported only when one of these names is first used, so that ``dsc-uci`` and the
# standalone engines answer ``uci`` immediately (a chess GUI's start-up limit is a few seconds,
# and the first launch of the inference libraries on a computer can take much longer).
_LAZY = {
    "load_model": ".checkpoint",
    "save_model": ".checkpoint",
    "DSCConfig": ".model",
    "DSCModel": ".model",
    "DSC": ".predictor",
    "WDL": ".predictor",
    "PositionInput": ".predictor",
    "Prediction": ".predictor",
    "ResolvedInputs": ".predictor",
}


def __getattr__(name: str):
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "DEFAULT_ELO",
    "DSC",
    "DSCConfig",
    "DSCModel",
    "MODELS",
    "ModelEntry",
    "PositionInput",
    "Prediction",
    "ResolvedInputs",
    "WDL",
    "__version__",
    "load_model",
    "resolve",
    "save_model",
]
