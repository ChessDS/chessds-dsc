# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Drive the UCI engine from python-chess (the Maia-3 README example with the executable and model
changed), then ask for the three most likely moves.

    python examples/uci_python_chess.py
    python examples/uci_python_chess.py --checkpoint-path dsc-23m.safetensors
"""

from __future__ import annotations

import argparse
import sys

import chess
import chess.engine


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dsc-23m")
    p.add_argument("--checkpoint-path", default=None)
    a = p.parse_args()
    selection = (
        ["--checkpoint-path", a.checkpoint_path] if a.checkpoint_path else ["--model", a.model]
    )
    # "dsc-uci" works when the package's console script is on PATH; the module form always works.
    command = [
        sys.executable,
        "-m",
        "chessds_dsc.uci",
        *selection,
        "--use-uci-history",
        "--elo",
        "1500",
    ]

    eng = chess.engine.SimpleEngine.popen_uci(command)
    board = chess.Board()
    print(eng.play(board, limit=chess.engine.Limit(nodes=1)).move)

    board.push_san("e4")
    board.push_san("c5")
    eng.configure({"SelfElo": 2650, "OppoElo": 2600, "Temperature": 0})
    for info in eng.analyse(board, chess.engine.Limit(depth=1), multipv=3):
        wdl = info["wdl"].pov(board.turn)
        print(
            f"{info['multipv']}. {board.san(info['pv'][0]):6s} wdl {wdl.wins}/{wdl.draws}/{wdl.losses}"
        )
    eng.close()


if __name__ == "__main__":
    main()
