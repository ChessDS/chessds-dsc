"""The UCI engine, driven the way GUIs and python-chess drive it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import chess
import chess.engine
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]


def engine_command(weights_path: str, *extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "chessds_dsc.uci",
        "--checkpoint-path",
        weights_path,
        "--device",
        "cpu",
        *extra,
    ]


def env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(PKG_ROOT) + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    return e


@pytest.fixture(scope="module")
def engine(tiny_history_model_path):
    # The Maia-3 README example, with only the executable and model selection changed.
    eng = chess.engine.SimpleEngine.popen_uci(
        engine_command(tiny_history_model_path, "--use-uci-history", "--elo", "1500"), env=env()
    )
    yield eng
    eng.quit()


def test_play_both_colors_and_options(engine):
    assert engine.id["name"].startswith("ChessDS DSC")
    for name in ("Elo", "SelfElo", "OppoElo", "Temperature", "TopP", "MultiPV", "TimeControl"):
        assert name in engine.options
    assert engine.options["Elo"].default == 1500 and engine.options["MultiPV"].max == 20
    board = chess.Board()
    result = engine.play(board, chess.engine.Limit(nodes=1))
    assert result.move in board.legal_moves
    board.push(result.move)
    result = engine.play(board, chess.engine.Limit(nodes=1))  # black to move
    assert result.move in board.legal_moves
    fen_board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert engine.play(fen_board, chess.engine.Limit(time=0.1)).move in fen_board.legal_moves


def test_analyse_multipv_lines_and_deterministic_bestmove(engine):
    board = chess.Board()
    board.push_san("e4")
    infos = engine.analyse(board, chess.engine.Limit(depth=1), multipv=3)
    assert len(infos) == 3
    for rank, info in enumerate(infos, start=1):
        assert info["multipv"] == rank and info["depth"] == 1
        assert info["pv"][0] in board.legal_moves
        assert "score" in info and "wdl" in info
        wdl = info["wdl"].pov(board.turn)
        assert wdl.wins + wdl.draws + wdl.losses == 1000
    assert (
        infos[0]["score"].relative.score()
        == infos[0]["wdl"].relative.wins - infos[0]["wdl"].relative.losses
    )
    engine.configure({"Temperature": 0})
    moves = {engine.play(board, chess.engine.Limit(nodes=1)).move for _ in range(5)}
    assert moves == {infos[0]["pv"][0]}
    engine.configure({"Temperature": 1.0, "SelfElo": 2700, "OppoElo": 2300})
    assert engine.play(board, chess.engine.Limit(nodes=1)).move in board.legal_moves


def test_raw_protocol_transcript(tiny_strict_model_path):
    """What a GUI sends: uci, setoption, isready, ucinewgame, position, go with clocks, infinite/stop."""
    proc = subprocess.Popen(
        engine_command(tiny_strict_model_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env(),
    )

    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def read_until(token):
        lines = []
        while True:
            line = proc.stdout.readline()
            assert line, f"engine exited before {token!r}; got {lines}"
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    send("uci")
    lines = read_until("uciok")
    assert any(line.startswith("id name ChessDS DSC") for line in lines)
    assert any(line.startswith("option name MultiPV type spin") for line in lines)
    send("setoption name MultiPV value 2")
    send("setoption name Elo value 2400")
    send("isready")
    read_until("readyok")
    send("ucinewgame")
    send("isready")
    read_until("readyok")
    send("position startpos moves e2e4 e7e5 g1f3")
    send("go wtime 60000 btime 60000 winc 1000 binc 1000")
    lines = read_until("bestmove")
    infos = [line for line in lines if line.startswith("info depth")]
    assert len(infos) == 2 and all(
        " multipv " in line and " wdl " in line and " pv " in line for line in infos
    )
    best = lines[-1].split()[1]
    assert (
        chess.Move.from_uci(best)
        in chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2").legal_moves
    )
    send("position fen 8/8/8/8/8/4k3/8/4K2R w K - 0 1 moves e1g1")
    send("go infinite")
    lines = read_until("info depth 1 multipv 1")
    send("stop")
    lines = read_until("bestmove")
    assert (
        chess.Move.from_uci(lines[-1].split()[1])
        in chess.Board("8/8/8/8/8/4k3/8/5RK1 b - - 1 1").legal_moves
    )
    # A finished game: no legal moves -> bestmove 0000.
    send("position fen rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    send("go")
    lines = read_until("bestmove")
    assert lines[-1] == "bestmove 0000"
    send("quit")
    assert proc.wait(timeout=10) == 0


def test_list_models_and_cache_cli_run():
    out = subprocess.run(
        [sys.executable, "-m", "chessds_dsc.uci", "--list-models"],
        capture_output=True,
        text=True,
        env=env(),
    )
    assert out.returncode == 0
    out = subprocess.run(
        [sys.executable, "-c", "from chessds_dsc.cache_cli import main; main(['--list'])"],
        capture_output=True,
        text=True,
        env=env(),
    )
    assert out.returncode == 0


def test_searchmoves_restricts_bestmove_and_lines(engine):
    board = chess.Board()
    only = [chess.Move.from_uci("d2d4"), chess.Move.from_uci("c2c4")]
    for _ in range(6):
        assert engine.play(board, chess.engine.Limit(nodes=1), root_moves=only).move in only
    infos = engine.analyse(board, chess.engine.Limit(depth=1), multipv=5, root_moves=only)
    assert len(infos) == 2 and {i["pv"][0] for i in infos} == set(only)
    engine.configure({"Temperature": 0})
    best = engine.play(board, chess.engine.Limit(nodes=1), root_moves=only).move
    assert best == infos[0]["pv"][0]
    engine.configure({"Temperature": 1.0})
    # Unrestricted again afterwards.
    infos = engine.analyse(board, chess.engine.Limit(depth=1), multipv=5)
    assert len(infos) == 5


def test_go_ponder_holds_the_answer_until_ponderhit(tiny_strict_model_path):
    proc = subprocess.Popen(
        engine_command(tiny_strict_model_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env(),
    )

    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def read_until(token):
        lines = []
        while True:
            line = proc.stdout.readline()
            assert line, f"engine exited before {token!r}; got {lines}"
            lines.append(line.strip())
            if line.startswith(token):
                return lines

    send("uci")
    read_until("uciok")
    send("isready")
    read_until("readyok")
    send("position startpos moves e2e4")
    send("go ponder wtime 1000 btime 1000")
    lines = read_until("info depth 1 multipv 1")
    assert not any(line.startswith("bestmove") for line in lines)
    send("ponderhit")
    lines = read_until("bestmove")
    assert (
        chess.Move.from_uci(lines[-1].split()[1])
        in chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").legal_moves
    )
    # A restriction that leaves no legal move answers 0000 without info lines.
    send("position startpos")
    send("go searchmoves e2e5 a7a6")
    lines = read_until("bestmove")
    assert lines[-1] == "bestmove 0000" and not any(line.startswith("info") for line in lines)
    send("quit")
    assert proc.wait(timeout=10) == 0


def test_empty_searchmoves_gives_no_move(engine, tiny_strict_model_path):
    board = chess.Board()
    assert not engine.play(board, chess.engine.Limit(nodes=1), root_moves=[]).move  # null move
    assert engine.play(board, chess.engine.Limit(nodes=1)).move in board.legal_moves
    proc = subprocess.Popen(
        engine_command(tiny_strict_model_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env(),
    )
    proc.stdin.write("uci\nisready\nposition startpos\ngo searchmoves\nquit\n")
    proc.stdin.flush()
    out = proc.stdout.read().splitlines()
    assert proc.wait(timeout=10) == 0
    assert "bestmove 0000" in out and not any(line.startswith("info") for line in out)
