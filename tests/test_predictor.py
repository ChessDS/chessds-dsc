from __future__ import annotations

import chess
import numpy as np
import pytest
from chessds_dsc import DSC, PositionInput
from chessds_dsc.predictor import DEFAULT_ELO


@pytest.fixture(scope="module")
def hist(tiny_history_model_path):
    return DSC.load(checkpoint_path=tiny_history_model_path, device="cpu")


@pytest.fixture(scope="module")
def strict(tiny_strict_model_path):
    return DSC.load(checkpoint_path=tiny_strict_model_path, device="cpu")


def game(sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans.split():
        board.push_san(san)
    return board


def test_probabilities_over_legal_moves_and_defaults(hist):
    board = game("e4 c5 Nf3")
    pred = hist.predict(board)
    assert set(pred.moves) == {m.uci() for m in board.legal_moves}
    assert abs(sum(pred.moves.values()) - 1.0) < 1e-5
    values = list(pred.moves.values())
    assert values == sorted(values, reverse=True)
    assert pred.best == next(iter(pred.moves))
    assert abs(sum(pred.wdl) - 1.0) < 1e-5 and pred.wdl.win >= 0
    i = pred.inputs
    assert i.elo_mover == DEFAULT_ELO == i.elo_opponent
    assert (i.time_control, i.rating_system, i.venue) == ("classical", "fide", "otb")
    assert i.history_boards == 3 and i.repetition == (False, False)
    assert i.repetition_source == "move_stack" and i.fen == board.fen()


def test_ratings_are_assigned_by_color(hist):
    board = game("e4 e5")  # white to move
    a = hist.predict(board, white_elo=2800, black_elo=2000)
    assert (a.inputs.elo_mover, a.inputs.elo_opponent) == (2800, 2000)
    board.push_san("Nf3")  # black to move
    b = hist.predict(board, white_elo=2800, black_elo=2000)
    assert (b.inputs.elo_mover, b.inputs.elo_opponent) == (2000, 2800)
    assert hist.predict(board, white_elo=2800, black_elo=2000).moves != hist.predict(board).moves


def test_history_behaviour(hist, strict):
    board = game("e4 e5 Nf3 Nf6 Nc3 Nc6")
    fen_only = chess.Board(board.fen())
    with_hist = hist.predict(board)
    without = hist.predict(fen_only)
    forced_off = hist.predict(board, use_history=False)
    assert with_hist.inputs.history_boards == 6 and without.inputs.history_boards == 0
    assert with_hist.moves != without.moves
    assert forced_off.moves == without.moves  # same tokens: current board in every slot
    assert without.inputs.repetition_source == "assumed_none"
    with pytest.raises(ValueError, match="use_history=True"):
        hist.predict(fen_only, use_history=True)
    # Strict models never read history and are path-invariant.
    s1, s2 = strict.predict(board), strict.predict(fen_only)
    assert s1.inputs.history_boards == 0 and s1.moves == s2.moves
    other_path = game("e4 e5 Nc3 Nc6 Nf3 Nf6")
    assert strict.predict(other_path).moves == s1.moves
    assert strict.predict(board, use_history=True).moves == s1.moves
    assert strict.predict(board, white_elo=2200).moves != s1.moves  # ratings still matter


def test_repetition_flags(hist):
    board = game("Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8")
    replayed = hist.predict(board, use_history=False)
    assert replayed.inputs.repetition == (True, True)
    fen_only = chess.Board(board.fen())
    assumed = hist.predict(fen_only)
    assert assumed.inputs.repetition == (False, False) and assumed.moves != replayed.moves
    explicit = hist.predict(fen_only, repetition=(True, True))
    assert explicit.inputs.repetition_source == "explicit" and explicit.moves == replayed.moves
    partial = chess.Board(board.fen())
    partial.push_san("Nf3")
    assert hist.predict(partial).inputs.repetition_source == "assumed_none"


def test_terminal_position_and_validation(hist):
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    pred = hist.predict(mate)
    assert pred.moves == {} and pred.best is None and abs(sum(pred.wdl) - 1) < 1e-5
    with pytest.raises(ValueError, match="time_control"):
        hist.predict(chess.Board(), time_control="correspondence")
    with pytest.raises(ValueError, match="rating_system"):
        hist.predict(chess.Board(), rating_system="unknown")
    with pytest.raises(ValueError, match="ratings"):
        hist.predict(chess.Board(), white_elo=-5)
    pred = hist.predict(chess.Board(), rating_system="lichess", time_control="blitz")
    assert pred.inputs.venue == "online"
    assert hist.predict(chess.Board(), rating_system="lichess", venue="otb").inputs.venue == "otb"


def test_batch_matches_single_and_accepts_fen_strings(hist):
    boards = [game("e4"), game("e4 e5 Nf3 Nc6 Bb5"), chess.Board()]
    items = [PositionInput(b, white_elo=2500 + 10 * i) for i, b in enumerate(boards)]
    batch = hist.predict_batch(items)
    for item, pred in zip(items, batch, strict=True):
        single = hist.predict(item.board, white_elo=item.white_elo)
        assert single.moves.keys() == pred.moves.keys()
        assert np.allclose(list(single.moves.values()), list(pred.moves.values()), atol=1e-6)
    from_fen = hist.predict(boards[1].fen())
    assert from_fen.inputs.history_boards == 0
    assert hist.predict_batch([]) == []
