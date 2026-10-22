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

from .checkpoint import load_model, save_model
from .model import DSCConfig, DSCModel
from .predictor import DEFAULT_ELO, DSC, WDL, PositionInput, Prediction, ResolvedInputs
from .registry import MODELS, ModelEntry, resolve

__version__ = "0.1.0"

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
