# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Predict human moves with the Python API: a FEN-only position and a position with its game.

python examples/python_api.py                       # downloads dsc-23m and dsc-23m-fen
python examples/python_api.py --checkpoint-path ... # a local weights file instead
"""

from __future__ import annotations

import argparse

import chess
from chessds_dsc import DSC


def show(title: str, pred) -> None:
    print(f"\n{title}")
    i = pred.inputs
    print(
        f"  inputs: mover {i.elo_mover:.0f} vs {i.elo_opponent:.0f}, {i.time_control}, "
        f"{i.rating_system}/{i.venue}, previous positions {i.history_boards}, "
        f"repetition {i.repetition} ({i.repetition_source})"
    )
    for uci, prob in list(pred.moves.items())[:5]:
        print(f"  {uci:6s} {prob:6.1%}")
    print(
        f"  WDL for the side to move (auxiliary head): {pred.wdl.win:.2f} / {pred.wdl.draw:.2f} / {pred.wdl.loss:.2f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dsc-23m")
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    model = DSC.load(a.model, checkpoint_path=a.checkpoint_path, device=a.device)
    print(f"loaded {model.model_id} ({model.mode}) on {model.device}")

    # 1. A position with its whole game: history and repetition state come from the move stack.
    board = chess.Board()
    for san in "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6".split():
        board.push_san(san)
    show(
        "Najdorf after 7...Be6, 2700 vs 2650, classical FIDE",
        model.predict(board, white_elo=2700, black_elo=2650),
    )

    # 2. The same position from a FEN only: no previous positions, repetition assumed absent.
    show("Same position from its FEN", model.predict(board.fen(), white_elo=2700, black_elo=2650))

    # 3. Defaults: both players at 2578 (the reference strength) when ratings are not given.
    show("Start position, default ratings", model.predict(chess.Board()))

    # 4. Lichess blitz: tell the model which rating scale and time control apply.
    show(
        "Start position as 2300 vs 2300 Lichess blitz",
        model.predict(
            chess.Board(),
            white_elo=2300,
            black_elo=2300,
            rating_system="lichess",
            time_control="blitz",
        ),
    )


if __name__ == "__main__":
    main()
