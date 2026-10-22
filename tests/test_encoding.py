from __future__ import annotations

import chess
import numpy as np
import pytest
from chessds_dsc import encoding as E
from conftest import boards_after

# White: pawn b7 promotes on b8 (push) and a8 (capture); both sides may castle both ways.
PROMO_WHITE = "r3k2r/1P6/8/8/8/8/8/R3K2R w KQkq - 0 1"
PROMO_BLACK = "r3k2r/8/8/8/8/8/1p6/R3K2R b KQkq - 0 1"


def test_layout_constants():
    assert E.TOKEN_DIM == 104 and E.N_MOVES == 16384
    assert E.C_CASTLE == 96 and E.C_EP == 100 and E.C_REP == 101 and E.C_HALFMOVE == 103
    assert E.N_BOARDS == 8 and E.N_HISTORY == 7


def test_public_choices_exclude_untrained_codes():
    assert "unknown" not in E.TIME_CONTROLS and "correspondence" not in E.TIME_CONTROLS
    assert "unknown" not in E.RATING_SYSTEMS and "unknown" not in E.VENUES
    assert set(E.VENUE_FOR_RATING_SYSTEM) == set(E.RATING_SYSTEMS)
    assert set(E.VENUE_FOR_RATING_SYSTEM.values()) <= set(E.VENUES)


@pytest.mark.parametrize("fen", [PROMO_WHITE, PROMO_BLACK])
def test_move_index_round_trip_all_legal_moves(fen):
    board = chess.Board(fen)
    white = board.turn == chess.WHITE
    legal = list(board.legal_moves)
    assert {m.promotion for m in legal if m.promotion} == {
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
    }
    assert any(board.is_kingside_castling(m) for m in legal)
    seen = set()
    for move in legal:
        idx = E.move_index(move, white)
        assert 0 <= idx < E.N_MOVES and idx not in seen
        seen.add(idx)
        assert E.index_to_move(idx, board) == move


def test_move_index_layout_and_black_flip():
    assert E.move_index(chess.Move(chess.E1, chess.G1), True) == (chess.E1 * 64 + chess.G1) * 4
    assert (
        E.move_index(chess.Move(chess.B7, chess.B8, promotion=chess.KNIGHT), True)
        == (chess.B7 * 64 + chess.B8) * 4 + 3
    )
    # Black's e8g8 is white's e1g1 in mover orientation; b2b1=N is b7b8=N.
    assert E.move_index(chess.Move(chess.E8, chess.G8), False) == E.move_index(
        chess.Move(chess.E1, chess.G1), True
    )
    assert (
        E.move_index(chess.Move(chess.B2, chess.B1, promotion=chess.KNIGHT), False)
        == (chess.B7 * 64 + chess.B8) * 4 + 3
    )
    board = chess.Board(PROMO_WHITE)
    assert E.index_to_move((chess.B7 * 64 + chess.B8) * 4, board).promotion == chess.QUEEN


def test_legal_mask_and_indices_agree():
    for fen in (chess.STARTING_FEN, PROMO_WHITE, PROMO_BLACK):
        board = chess.Board(fen)
        mask = E.legal_mask(board)
        pairs = E.legal_indices(board)
        assert mask.sum() == len(pairs) == board.legal_moves.count()
        assert {i for _, i in pairs} == set(np.flatnonzero(mask).tolist())
        assert {u for u, _ in pairs} == {m.uci() for m in board.legal_moves}


def test_piece_planes_orientation():
    board = chess.Board()
    white = E.piece_planes(board, True)
    black = E.piece_planes(board, False)
    # Own pawns on the second rank (squares 8..15) in both orientations.
    assert white[8:16, 0].all() and black[8:16, 0].all()
    assert white[:, :6].sum() == 16 and white[:, 6:].sum() == 16
    board.push_san("e4")
    planes = E.piece_planes(board, False)  # black to move: white's pawn on e4 is an opponent pawn
    assert planes[E.orient(chess.E4, False), 6] == 1


def test_history_padding_and_limit():
    boards = boards_after("e4 e5 Nf3")
    tokens = E.encode_position(boards)
    assert E.hist_count_of(boards) == 3
    # Slots 3..7 repeat the earliest board (the start position).
    for k in range(4, E.N_BOARDS):
        assert (tokens[:, k * 12 : (k + 1) * 12] == tokens[:, 36:48]).all()
    long = boards_after("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5")
    assert E.hist_count_of(long) == 7
    assert E.encode_position(long).tobytes() == E.encode_position(long[:8]).tobytes()
    single = E.encode_position([boards[0]])
    assert all(
        (single[:, k * 12 : (k + 1) * 12] == single[:, :12]).all() for k in range(E.N_BOARDS)
    )


def test_history_boards_matches_manual_replay():
    board = chess.Board()
    for san in "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+".split():
        board.push_san(san)
    hist = E.history_boards(board)
    assert len(hist) == E.N_BOARDS and hist[0].fen() == board.fen()
    manual = boards_after("d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+")
    assert [b.board_fen() for b in hist] == [b.board_fen() for b in manual[: E.N_BOARDS]]
    assert all(not b.move_stack for b in hist)
    assert len(E.history_boards(chess.Board())) == 1


def test_castling_en_passant_halfmove_channels():
    board = chess.Board(PROMO_BLACK)
    tokens = E.encode_position([board])
    assert (tokens[:, E.C_CASTLE : E.C_CASTLE + 4] == 1).all()
    board = chess.Board()
    for san in "e4 a6 e5 d5".split():
        board.push_san(san)
    tokens = E.encode_position([board])
    assert tokens[:, E.C_EP].sum() == 1 and tokens[E.orient(chess.D6, True), E.C_EP] == 1
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 150 80")
    assert (E.encode_position([board])[:, E.C_HALFMOVE] == 100).all()
    binary = np.delete(E.encode_position([board]), E.C_HALFMOVE, axis=1)
    assert set(np.unique(binary).tolist()) <= {0, 1}


def test_repetition_flags_and_provenance():
    board = chess.Board()
    for san in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]:
        board.push_san(san)
    assert E.repetition_flags(board) == (True, True) and E.rule_state_known(board)
    from_stack = E.encode_position([board])
    assert from_stack[0, E.C_REP] == 1 and from_stack[0, E.C_REP + 1] == 1
    fen_only = chess.Board(board.fen())
    assert not E.rule_state_known(fen_only)
    assert E.encode_position([fen_only])[0, E.C_REP] == 0
    assert (E.encode_position([fen_only], repetition=(True, True)) == from_stack).all()
    partial = chess.Board(board.fen())
    partial.push_san("Nf3")
    assert not E.rule_state_known(partial)
    assert E.rule_state_known(chess.Board())


@pytest.mark.parametrize(
    "line_a, line_b",
    [
        ("e4 e5 Nf3 Nf6 Nc3 Nc6", "e4 e5 Nc3 Nc6 Nf3 Nf6"),
        ("e4 e5 Nf3 Nf6 Nc3", "e4 e5 Nc3 Nf6 Nf3"),
    ],
)
def test_no_history_encoding_is_path_invariant(line_a, line_b):
    a, b = boards_after(line_a), boards_after(line_b)
    assert a[0].board_fen() == b[0].board_fen()
    assert E.encode_position([a[0]]).tobytes() == E.encode_position([b[0]]).tobytes()
    assert E.encode_position(a).tobytes() != E.encode_position(b).tobytes()
