"""The likelihood output modes of the UCI engine: probabilities, ordering, and what is computed."""

from __future__ import annotations

import io
import re

import chess
import pytest
from chessds_dsc import DSC, uci
from chessds_dsc.uci import percent

P_LINE = re.compile(r"^info string (\S+) \(P: ([0-9.]+)%\)$")


def make_engine(path: str, *extra: str) -> uci.Engine:
    eng = uci.Engine(uci.parse_args(["--checkpoint-path", path, "--device", "cpu", *extra]))
    eng.log = lambda msg: None
    return eng


def run(eng: uci.Engine, *lines: str) -> list[str]:
    buf = io.StringIO()
    eng.out = buf
    eng.run([line + "\n" for line in lines])
    return buf.getvalue().splitlines()


def parse(lines: list[str]) -> dict:
    out = {"context": [], "p": [], "multipv": [], "summary": [], "bestmove": None}
    for line in lines:
        if m := P_LINE.match(line):
            out["p"].append((m.group(1), m.group(2)))
        elif line.startswith("info string DSC ") and "P = probability" in line:
            out["context"].append(line)
        elif line.startswith("info string DSC move probabilities"):
            out["summary"].append(line)
        elif line.startswith("info depth"):
            t = line.split()
            out["multipv"].append((int(t[t.index("multipv") + 1]), t[t.index("pv") + 1], line))
        elif line.startswith("bestmove"):
            out["bestmove"] = line.split()[1]
    return out


@pytest.fixture(scope="module")
def api_h(tiny_history_model_path):
    return DSC.load(checkpoint_path=tiny_history_model_path, device="cpu")


@pytest.fixture(scope="module")
def api_s(tiny_strict_model_path):
    return DSC.load(checkpoint_path=tiny_strict_model_path, device="cpu")


def board_after(*ucis: str, fen: str | None = None) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    for u in ucis:
        board.push_uci(u)
    return board


def test_probabilities_equal_the_api_ranked_with_one_network_call(tiny_history_model_path, api_h):
    eng = make_engine(
        tiny_history_model_path,
        "--use-uci-history",
        "--output-mode",
        "likelihood",
        "--temperature",
        "0",
    )
    moves = ["e2e4", "c7c5", "g1f3"]  # Black to move, three previous positions
    out = parse(
        run(
            eng,
            "setoption name MultiPV value 4",
            "position startpos moves " + " ".join(moves),
            "go",
        )
    )
    pred = api_h.predict(board_after(*moves))
    top = list(pred.moves.items())[:4]
    # Lc0 order (least probable first) on the P lines, best first on the multipv lines.
    assert out["p"] == [(u, percent(p)) for u, p in reversed(top)]
    assert [(r, m) for r, m, _ in out["multipv"]] == [(i + 1, u) for i, (u, _) in enumerate(top)]
    assert not any(" score " in line or " wdl " in line for _, _, line in out["multipv"])
    board = board_after(*moves)
    ranked = ", ".join(f"{board.san(chess.Move.from_uci(u))} {percent(p)}%" for u, p in top)
    assert len(out["summary"]) == 1 and ranked in out["summary"][0]
    assert f"4 of {len(pred.moves)} legal moves shown" in out["summary"][0]
    assert out["bestmove"] == top[0][0]
    assert (eng.forward_calls, eng.forward_positions) == (1, 1)


def test_context_line_reports_the_active_inputs(tiny_history_model_path):
    eng = make_engine(tiny_history_model_path, "--use-uci-history", "--output-mode", "likelihood")
    out = parse(
        run(
            eng,
            "setoption name SelfElo value 2700",
            "setoption name OppoElo value 2300",
            "setoption name TimeControl value blitz",
            "setoption name RatingSystem value lichess",
            "setoption name Temperature value 0.5",
            "position startpos moves e2e4 c7c5 g1f3",
            "go",
        )
    )
    (context,) = out["context"]
    # Black to move: SelfElo is Black's rating.
    assert "White 2300, Black 2700, blitz, lichess ratings" in context
    assert "3 previous positions" in context and "not seen before (from the move list)" in context
    assert "temperature 0.5, top-p 1" in context and "not an evaluation" in context
    out = parse(run(eng, f"position fen {chess.STARTING_FEN.replace(' w ', ' b ')}", "go"))
    assert "0 previous positions" in out["context"][0]
    assert "repetition unknown" in out["context"][0]


@pytest.mark.parametrize("mode", ["likelihood", "likelihood-wdl"])
def test_truncation_and_sampling_settings_leave_probabilities_alone(tiny_strict_model_path, mode):
    eng = make_engine(tiny_strict_model_path, "--output-mode", mode)
    pos = "position startpos moves d2d4"
    full = parse(
        run(eng, "setoption name MultiPV value 20", "setoption name Temperature value 0", pos, "go")
    )
    top = list(reversed(full["p"]))  # best first
    short = parse(run(eng, "setoption name MultiPV value 3", pos, "go"))
    assert list(reversed(short["p"])) == top[:3]  # not renormalised
    assert sum(float(v) for _, v in short["p"]) < 100
    assert short["bestmove"] == top[0][0]  # temperature 0: the most probable move
    hot = parse(
        run(
            eng,
            "setoption name Temperature value 2.5",
            "setoption name TopP value 0.6",
            pos,
            "go",
        )
    )
    assert list(reversed(hot["p"])) == top[:3]
    assert chess.Move.from_uci(hot["bestmove"]) in board_after("d2d4").legal_moves


def test_searchmoves_shows_unrestricted_probabilities(tiny_strict_model_path, api_s):
    eng = make_engine(tiny_strict_model_path, "--output-mode", "likelihood", "--temperature", "0")
    out = parse(run(eng, "position startpos", "go searchmoves d2d4 c2c4 a1a8"))
    pred = api_s.predict(chess.Board())
    allowed = [(u, p) for u, p in pred.moves.items() if u in {"d2d4", "c2c4"}]
    assert list(reversed(out["p"])) == [(u, percent(p)) for u, p in allowed]
    assert [m for _, m, _ in out["multipv"]] == [u for u, _ in allowed]
    assert out["bestmove"] == allowed[0][0]
    mass = percent(sum(p for _, p in allowed))
    assert f"searchmoves allows 2 of 20 moves, together {mass}%" in out["summary"][0]
    assert "rescaled to 100%" in out["summary"][0]


def test_likelihood_wdl_is_the_default_and_carries_the_candidates_wdl(
    tiny_strict_model_path, api_s
):
    eng = make_engine(tiny_strict_model_path, "--temperature", "0")  # default output mode
    assert eng.output_mode == "likelihood-wdl"
    out = parse(run(eng, "position startpos moves e2e4", "go"))
    board = board_after("e2e4")
    top = list(api_s.predict(board).moves.items())[:5]
    expected = []
    for rank, (u, _) in enumerate(top, start=1):
        child = board.copy()
        child.push_uci(u)
        w = api_s.predict(child).wdl  # the opponent's view after the move
        win, draw, loss = uci.permille((w.loss, w.draw, w.win))  # the mover's view
        expected.append(
            f"info depth 1 multipv {rank} score cp {win - loss} wdl {win} {draw} {loss} pv {u}"
        )
    assert [line for _, _, line in out["multipv"]] == expected
    assert out["p"] == [(u, percent(p)) for u, p in reversed(top)]
    assert out["bestmove"] == top[0][0]
    assert (eng.forward_calls, eng.forward_positions) == (2, 6)


def test_no_move_answers_stay_bare(tiny_strict_model_path):
    eng = make_engine(tiny_strict_model_path, "--output-mode", "likelihood")
    mate = "position startpos moves f2f3 e7e5 g2g4 d8h4"
    assert run(eng, mate, "go") == ["bestmove 0000"]
    assert run(eng, "position startpos", "go searchmoves a1a8") == ["bestmove 0000"]
    assert run(eng, "position startpos", "go searchmoves 0000") == ["bestmove 0000"]
    assert run(eng, "position startpos", "go searchmoves") == ["bestmove 0000"]


def test_infinite_and_ponder_hold_the_answer(tiny_history_model_path):
    eng = make_engine(tiny_history_model_path, "--use-uci-history", "--output-mode", "likelihood")
    lines = run(eng, "position startpos moves e2e4", "go infinite")
    assert lines and not any(line.startswith("bestmove") for line in lines)
    assert run(eng, "stop")[0].startswith("bestmove ")
    lines = run(eng, "position startpos moves e2e4 e7e5", "go ponder wtime 1000 btime 1000")
    assert not any(line.startswith("bestmove") for line in lines)
    assert run(eng, "ponderhit")[0].startswith("bestmove ")


def test_output_mode_option(tiny_strict_model_path):
    eng = make_engine(tiny_strict_model_path, "--output-mode", "likelihood")
    head = run(eng, "uci")
    assert (
        "option name OutputMode type combo default likelihood var likelihood "
        "var likelihood-wdl var probability-as-score" in head
    )
    run(eng, "setoption name OutputMode value likelihood-wdl")
    out = parse(run(eng, "position startpos", "go"))
    assert all(" score cp " in line and " wdl " in line for _, _, line in out["multipv"])
    run(eng, "setoption name OutputMode value maia")  # no longer a mode: ignored
    assert eng.output_mode == "likelihood-wdl"


def test_probability_as_score_puts_the_probability_in_the_score(tiny_history_model_path, api_h):
    eng = make_engine(
        tiny_history_model_path,
        "--use-uci-history",
        "--temperature",
        "0",
        "--output-mode",
        "probability-as-score",
    )
    # Black to move: sent negative, so that GUIs showing White's point of view print it positive.
    moves = ["e2e4", "c7c5", "g1f3"]
    out = parse(run(eng, "position startpos moves " + " ".join(moves), "go"))
    top = list(api_h.predict(board_after(*moves)).moves.items())[:5]
    scores = [(m, int(line.split(" score cp ")[1].split()[0])) for _, m, line in out["multipv"]]
    assert scores == [(u, -round(p * uci.SCORE_PER_PROBABILITY)) for u, p in top]
    assert not any(" wdl " in line for _, _, line in out["multipv"])
    assert out["p"] == [(u, percent(p)) for u, p in reversed(top)]
    assert "score = probability x 10000" in out["context"][0]
    assert out["bestmove"] == top[0][0]
    assert (eng.forward_calls, eng.forward_positions) == (1, 1)
    # White to move, with searchmoves: positive, the unrestricted probabilities.
    out = parse(run(eng, "position startpos", "go searchmoves d2d4 c2c4"))
    pred = api_h.predict(chess.Board())
    allowed = [(u, p) for u, p in pred.moves.items() if u in {"d2d4", "c2c4"}]
    got = [int(line.split(" score cp ")[1].split()[0]) for _, _, line in out["multipv"]]
    assert got == [round(p * uci.SCORE_PER_PROBABILITY) for _, p in allowed]
    # No move to give: still a bare bestmove.
    assert run(eng, "position startpos moves f2f3 e7e5 g2g4 d8h4", "go") == ["bestmove 0000"]
