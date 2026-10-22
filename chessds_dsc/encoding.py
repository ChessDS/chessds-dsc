# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Board and move encoding for DSC models.

Everything is oriented to the side to move (the "mover"): a square's index is its chess index
when white is to move and the vertically flipped index (``square ^ 56``) when black is to move,
and "own" pieces are the mover's. Each of the 64 squares becomes one token of ``TOKEN_DIM`` = 104
uint8 channels:

    [0, 96)    piece one-hot for 8 boards x 12 pieces. Board 0 is the current position, boards
               1..7 are the positions 1..7 plies earlier, all oriented to the *current* mover.
               Each board's 12 channels are own P N B R Q K, then opponent P N B R Q K. Missing
               history repeats the earliest board available. A single board fills all 8 slots:
               that is the no-history encoding the strict models always use.
    [96, 100)  castling rights, the same on every square: own K-side, own Q-side, opp K, opp Q
    100        1 on the square of a legal en-passant capture, else 0
    [101, 103) repetition, the same on every square: position seen >= 1 time before, >= 2 times
    103        min(halfmove clock, 100); the model scales it by 1/100

Move index space (``N_MOVES`` = 16384): ``(from * 64 + to) * 4 + promo`` in mover orientation,
with promo 0 = none or queen (a pawn reaching the last rank always promotes, so channel 0 is
unambiguous), 1 = rook, 2 = bishop, 3 = knight.

Conditioning codes (``TIME_CONTROL_CODES`` etc.) are the integer inputs of the model's
embedding tables. Code 0 ("unknown") exists in each table but never occurred in training; the
public API does not offer it.
"""

from __future__ import annotations

from collections.abc import Sequence

import chess
import numpy as np

ENCODING_VERSION = "dsc-encoding-v1"

N_SQUARES = 64
N_HISTORY = 7
N_BOARDS = N_HISTORY + 1
N_PIECE_CHANNELS = 12
PIECE_DIM = N_BOARDS * N_PIECE_CHANNELS  # 96
C_CASTLE = PIECE_DIM  # 4 channels
C_EP = PIECE_DIM + 4
C_REP = PIECE_DIM + 5  # 2 channels
C_HALFMOVE = PIECE_DIM + 7
TOKEN_DIM = PIECE_DIM + 8  # 104

N_PROMO = 4
N_MOVES = N_SQUARES * N_SQUARES * N_PROMO  # 16384

TIME_CONTROL_CODES = {
    "unknown": 0,
    "bullet": 1,
    "blitz": 2,
    "rapid": 3,
    "classical": 4,
    "correspondence": 5,
}
RATING_SYSTEM_CODES = {"unknown": 0, "fide": 1, "lichess": 2, "other": 3}
VENUE_CODES = {"unknown": 0, "otb": 1, "online": 2}
N_TIME_CONTROLS = len(TIME_CONTROL_CODES)
N_RATING_SYSTEMS = len(RATING_SYSTEM_CODES)
N_VENUES = len(VENUE_CODES)

# The values the public API accepts: every code that occurred in training. "correspondence"
# and the "unknown" codes have embedding rows but never occurred, so they are not offered.
TIME_CONTROLS = ("bullet", "blitz", "rapid", "classical")
RATING_SYSTEMS = ("fide", "lichess", "other")
VENUES = ("otb", "online")
# In the training data the venue was fully determined by the rating system.
VENUE_FOR_RATING_SYSTEM = {"fide": "otb", "lichess": "online", "other": "online"}

_PROMO_CODE = {None: 0, chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}
_PROMO_PIECE = {1: chess.ROOK, 2: chess.BISHOP, 3: chess.KNIGHT}
_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


# ── squares and moves ───────────────────────────────────────────────────────


def orient(square: int, white_to_move: bool) -> int:
    """Square index from the mover's point of view (vertical flip when black is to move)."""
    return square if white_to_move else square ^ 56


def move_index(move: chess.Move, white_to_move: bool) -> int:
    """Policy index of ``move`` in mover orientation, in ``[0, N_MOVES)``."""
    f = orient(move.from_square, white_to_move)
    t = orient(move.to_square, white_to_move)
    return (f * N_SQUARES + t) * N_PROMO + _PROMO_CODE[move.promotion]


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of :func:`move_index` on ``board`` (channel 0 on a pawn reaching the last rank
    decodes as a queen promotion)."""
    white = board.turn == chess.WHITE
    promo_code = index % N_PROMO
    ft = index // N_PROMO
    f = orient(ft // N_SQUARES, white)
    t = orient(ft % N_SQUARES, white)
    promo = _PROMO_PIECE.get(promo_code)
    if promo is None:
        piece = board.piece_at(f)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(t) in (0, 7):
            promo = chess.QUEEN
    return chess.Move(f, t, promotion=promo)


def legal_mask(board: chess.Board) -> np.ndarray:
    """``(N_MOVES,)`` bool mask of the legal moves of ``board`` in mover orientation."""
    mask = np.zeros(N_MOVES, dtype=bool)
    white = board.turn == chess.WHITE
    for move in board.legal_moves:
        mask[move_index(move, white)] = True
    return mask


def legal_indices(board: chess.Board) -> list[tuple[str, int]]:
    """``[(uci, policy index)]`` for every legal move, in python-chess generation order."""
    white = board.turn == chess.WHITE
    return [(move.uci(), move_index(move, white)) for move in board.legal_moves]


# ── boards ──────────────────────────────────────────────────────────────────


def _bits(bitboard: int) -> np.ndarray:
    """``(64,)`` uint8 0/1 from a python-chess bitboard; element i is square i."""
    raw = np.frombuffer(bitboard.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")


def piece_planes(board: chess.Board, white_to_move: bool) -> np.ndarray:
    """``(64, 12)`` uint8 one-hot of the pieces, oriented to the mover given by ``white_to_move``;
    the mover's pieces fill columns 0-5."""
    mover = chess.WHITE if white_to_move else chess.BLACK
    columns = [
        _bits(board.pieces_mask(piece_type, color))
        for color in (mover, not mover)
        for piece_type in _PIECE_TYPES
    ]
    planes = np.stack(columns, axis=1)
    if not white_to_move:
        planes = planes.reshape(8, 8, N_PIECE_CHANNELS)[::-1].reshape(N_SQUARES, N_PIECE_CHANNELS)
    return planes


def encode_position(
    boards: Sequence[chess.Board], repetition: tuple[bool, bool] | None = None
) -> np.ndarray:
    """``(64, TOKEN_DIM)`` uint8 tokens.

    ``boards[0]`` is the current position and ``boards[1:]`` the previous positions, most recent
    first; at most ``N_HISTORY`` of them are used and missing history repeats the earliest board
    given. Pass a single board for the no-history encoding.

    The repetition channels describe the current position. When ``repetition`` is None they are
    read from ``boards[0]``'s move stack, which is only complete when the board was reached by
    replaying the whole game (see :func:`rule_state_known`); a board built from a FEN has an empty
    stack and encodes "no repetition". Callers that know the flags pass
    ``repetition=(seen_once_before, seen_twice_before)``.
    """
    current = boards[0]
    white = current.turn == chess.WHITE
    out = np.zeros((N_SQUARES, TOKEN_DIM), dtype=np.uint8)
    history = list(boards[:N_BOARDS])
    while len(history) < N_BOARDS:
        history.append(history[-1])
    for k, board in enumerate(history):
        out[:, k * N_PIECE_CHANNELS : (k + 1) * N_PIECE_CHANNELS] = piece_planes(board, white)
    mover = chess.WHITE if white else chess.BLACK
    out[:, C_CASTLE + 0] = current.has_kingside_castling_rights(mover)
    out[:, C_CASTLE + 1] = current.has_queenside_castling_rights(mover)
    out[:, C_CASTLE + 2] = current.has_kingside_castling_rights(not mover)
    out[:, C_CASTLE + 3] = current.has_queenside_castling_rights(not mover)
    if current.ep_square is not None and current.has_legal_en_passant():
        out[orient(current.ep_square, white), C_EP] = 1
    if repetition is None:
        repetition = repetition_flags(current)
    out[:, C_REP + 0] = repetition[0]
    out[:, C_REP + 1] = repetition[1]
    out[:, C_HALFMOVE] = min(current.halfmove_clock, 100)
    return out


def repetition_flags(board: chess.Board) -> tuple[bool, bool]:
    """``(seen >= 1 time before, seen >= 2 times before)`` from the board's move stack."""
    return bool(board.is_repetition(2)), bool(board.is_repetition(3))


def rule_state_known(board: chess.Board) -> bool:
    """True when the board's move stack covers the whole game from the standard start position,
    so that its repetition flags are complete. A board from a mid-game FEN, with or without moves
    pushed afterwards, returns False."""
    if board.root().fen() != chess.STARTING_FEN:
        return False
    return len(board.move_stack) == board.ply()


def history_boards(board: chess.Board) -> list[chess.Board]:
    """``[current, 1 ply ago, 2 plies ago, ...]`` from the board's move stack, at most ``N_BOARDS``
    boards. The copies carry no move stacks; read rule state from ``board`` itself."""
    boards = [board.copy(stack=False)]
    walker = board.copy()
    while len(boards) < N_BOARDS and walker.move_stack:
        walker.pop()
        boards.append(walker.copy(stack=False))
    return boards


def hist_count_of(boards: Sequence[chess.Board]) -> int:
    """Number of real previous boards in a :func:`encode_position` input, ``0..N_HISTORY``."""
    return min(len(boards) - 1, N_HISTORY)
