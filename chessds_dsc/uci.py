# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""``dsc-uci``: run a DSC model as a UCI chess engine.

The engine answers ``go`` with ``info`` lines for the most probable human moves and a
``bestmove`` chosen by sampling (``Temperature`` 0 picks the most probable move). The interface
follows the Maia-3 engine's conventions so that existing integrations only need to change the
executable and the model name:

* options ``Elo`` (both players), ``SelfElo`` (side to move), ``OppoElo``, ``Temperature``,
  ``TopP`` and ``MultiPV``; DSC adds ``TimeControl``, ``RatingSystem`` and ``OutputMode``;
* ``--use-uci-history`` rebuilds the previous positions from ``position ... moves ...``; without
  it the current position fills every history slot (FEN models never use history);
* ``depth`` is always 1: there is no search.

``OutputMode`` (``--output-mode``) chooses what ``go`` reports for the ``MultiPV`` most probable
moves, which always come first in ``multipv`` order:

* ``likelihood``: the move probabilities themselves, from one network call. A context line
  (``info string DSC ...``: ratings, time control, rating system, previous positions, repetition
  state, sampling settings), one ``info string <move> (P: <percent>%)`` line per move in the form
  of Lc0's verbose move statistics, least probable first because GUIs that read them (Nibbler)
  rank moves by arrival, then ``info depth 1 multipv N pv <move>`` lines without a score, and a
  summary line listing the probabilities highest first. No score or WDL is reported.
* ``likelihood-wdl`` (this command's default): as ``likelihood``, and each ``multipv`` line also
  carries the candidate's ``score cp`` (win minus loss, in permille, from the auxiliary neural
  WDL head after the move) and ``wdl`` in permille, exactly the lines of the Maia-3 engine; a GUI
  that shows only the score therefore looks as it does with Maia-3. One network call for the
  position, one for the candidates.
* ``probability-as-score``: opt-in, for GUIs that show nothing but the score (HIARCS Chess
  Explorer). As ``likelihood``, but each ``multipv`` line carries ``score cp`` = the move's
  probability x 10000, so a GUI that prints centipawns as pawns shows +41.73 for a 41.73% move.
  The sign is chosen for GUIs that show scores from White's point of view (HIARCS, ChessBase):
  negative when Black is to move, so that they print every probability as a positive number.
  The score is then not an evaluation: evaluation graphs, annotations and adjudication in that
  GUI read it as one, so use this mode for viewing only. No ``wdl`` is sent; one network call.

The probabilities shown are always the model's distribution over all legal moves: ``MultiPV``
only truncates the list, and ``Temperature`` / ``TopP`` only affect the move played. With
``go searchmoves`` the allowed moves are shown with their unrestricted probabilities; the move
played is drawn from them rescaled to sum to one.

The default rating is 2578 for both players (Maia-3 uses 1500). Repetition state is read from
the move list whenever ``position startpos moves ...`` supplies the whole game, with or without
``--use-uci-history``; after a ``position fen ...`` it is assumed absent.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import chess

from .constants import (
    DEFAULT_ELO,
    DEFAULT_RATING_SYSTEM,
    DEFAULT_TIME_CONTROL,
    RATING_SYSTEMS,
    TIME_CONTROLS,
)
from .registry import MODELS, format_model_list, resolve

if TYPE_CHECKING:
    from .predictor import DSC, PositionInput, Prediction

# torch, numpy and safetensors are imported only when the model is needed (``ensure_loaded``),
# never at import time: ``uci`` and ``isready`` are answered from the standard library and
# python-chess alone. On a computer that runs the engine for the first time, loading the
# inference libraries can take 10-20 s, longer than chess GUIs wait for ``uciok``.

ENGINE_NAME = "ChessDS DSC"
ENGINE_AUTHOR = "Albert W. Hamood (ChessDS)"
DEFAULT_MODEL = "dsc-23m"
MAX_MULTIPV = 20
OUTPUT_MODES = ("likelihood", "likelihood-wdl", "probability-as-score")
CANDIDATE_WDL_MODES = ("likelihood-wdl",)  # the modes that evaluate the candidates
SCORE_PER_PROBABILITY = 10_000  # probability-as-score: 41.73% -> score cp 4173, shown +41.73
DEFAULT_OUTPUT_MODE = "likelihood-wdl"
GO_KEYWORDS = {
    "searchmoves",
    "ponder",
    "wtime",
    "btime",
    "winc",
    "binc",
    "movestogo",
    "depth",
    "nodes",
    "mate",
    "movetime",
    "infinite",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dsc-uci",
        description="Run a ChessDS DSC model as a UCI chess engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", default=None, help=f"registered model id or alias (default {DEFAULT_MODEL})"
    )
    p.add_argument(
        "--checkpoint-path",
        "--checkpoint",
        dest="checkpoint_path",
        default=None,
        help="local .safetensors file (or a directory holding one) in the DSC format",
    )
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")
    p.add_argument("--seed", type=int, default=42, help="seed of the move sampler")
    p.add_argument("--elo", type=float, default=DEFAULT_ELO, help="default rating of both players")
    p.add_argument(
        "--temperature", type=float, default=1.0, help="sampling temperature; 0 = best move"
    )
    p.add_argument(
        "--top-p",
        "--top_p",
        dest="top_p",
        type=float,
        default=1.0,
        help="nucleus sampling threshold; 1.0 = off",
    )
    p.add_argument("--multipv", type=int, default=5, help="number of info lines per go")
    p.add_argument(
        "--output-mode",
        default=DEFAULT_OUTPUT_MODE,
        choices=OUTPUT_MODES,
        help="likelihood: move probabilities only (one network call); likelihood-wdl: "
        "probabilities and the candidates' auxiliary WDL as score cp / wdl (Maia-3 lines); "
        "probability-as-score: probabilities also sent as the score (x10000), for GUIs that "
        "show only the score; not an evaluation",
    )
    p.add_argument(
        "--use-uci-history",
        "--use_uci_history",
        dest="use_uci_history",
        action="store_true",
        help="rebuild the previous positions from 'position ... moves' (history models)",
    )
    p.add_argument("--time-control", default=DEFAULT_TIME_CONTROL, choices=TIME_CONTROLS)
    p.add_argument("--rating-system", default=DEFAULT_RATING_SYSTEM, choices=RATING_SYSTEMS)
    p.add_argument("--cache-dir", default=None, help="where downloaded models are stored")
    p.add_argument(
        "--base-url", default=None, help="download location (overrides CHESSDS_DSC_BASE_URL)"
    )
    p.add_argument("--local-files-only", action="store_true", help="never download")
    p.add_argument("--force-download", action="store_true", help="re-download the model files")
    p.add_argument("--list-models", action="store_true", help="list the registered models and exit")
    args = p.parse_args(argv)
    if args.list_models:
        print(format_model_list())
        raise SystemExit(0)
    if args.model is None and args.checkpoint_path is None:
        args.model = DEFAULT_MODEL
    if args.model is not None:
        try:
            args.model = resolve(args.model).id
        except KeyError:
            pass  # Preserve the normal error path for unknown or custom local model ids.
    return args


def percent(p: float) -> str:
    """A probability as a percentage string: two decimals, or two significant digits below
    0.01% so that a small probability never reads as zero."""
    value = 100.0 * float(p)
    if value >= 0.01:
        return f"{value:.2f}"
    import numpy as np

    return np.format_float_positional(value, precision=2, fractional=False, trim="-")


@dataclass
class Analysis:
    """One ``go``: the move played and what is shown."""

    chosen: str
    prediction: Prediction  # the unrestricted distribution over the legal moves
    shown: list[tuple[str, float]]  # up to MultiPV (move, unrestricted probability), best first
    allowed: set[str] | None = None  # go searchmoves restriction, if any
    allowed_mass: float = 1.0  # unrestricted probability of the allowed moves
    n_allowed: int = 0
    wdl: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # mover's view


def permille(values) -> tuple[int, ...]:
    """Probabilities -> integers summing to 1000 (largest remainders rounded up)."""
    scaled = [max(0.0, float(v)) * 1000 for v in values]
    ints = [int(v) for v in scaled]
    remainder = 1000 - sum(ints)
    order = sorted(range(len(scaled)), key=lambda i: scaled[i] - ints[i], reverse=True)
    for i in order[: max(0, remainder)]:
        ints[i] += 1
    return tuple(ints)


class Engine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dsc: DSC | None = None
        self.board = chess.Board()
        self.self_elo = float(args.elo)
        self.oppo_elo = float(args.elo)
        self.temperature = float(args.temperature)
        self.top_p = float(args.top_p)
        self.multipv = self._clamp_multipv(args.multipv)
        self.time_control = args.time_control
        self.rating_system = args.rating_system
        self.output_mode = args.output_mode
        self.forward_calls = 0  # network calls (predict_batch) and positions evaluated
        self.forward_positions = 0
        self.rng = None  # numpy Generator, created with the model (numpy is imported then)
        self._load_lock = threading.Lock()
        self._preload: threading.Thread | None = None
        self.pending_bestmove: str | None = None
        self.pending_search = False
        self.out = sys.stdout

    @staticmethod
    def _clamp_multipv(value) -> int:
        return min(MAX_MULTIPV, max(1, int(value)))

    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    def send(self, line: str) -> None:
        print(line, file=self.out, flush=True)

    # ── model ───────────────────────────────────────────────────────────────

    def ensure_loaded(self) -> DSC:
        """The loaded model; loads it (importing torch) on first use.

        Serialised with the background preload started by ``start_preload``: a ``go`` that
        arrives while the preload is running waits for it instead of loading twice.
        """
        with self._load_lock:
            if self.dsc is None:
                import numpy as np
                import torch

                from .predictor import DSC

                a = self.args
                what = a.checkpoint_path or a.model
                self.log(f"loading DSC model {what}")
                self.rng = np.random.default_rng(a.seed)
                torch.manual_seed(a.seed)
                self.dsc = DSC.load(
                    a.model or DEFAULT_MODEL,
                    checkpoint_path=a.checkpoint_path,
                    device=a.device,
                    cache_dir=a.cache_dir,
                    base_url=a.base_url,
                    local_files_only=a.local_files_only,
                    force_download=a.force_download,
                )
                variant = "FEN" if self.dsc.mode == "strict" else "main"
                self.log(f"{self._name()} ready ({variant}, {self.dsc.device})")
            return self.dsc

    def start_preload(self) -> None:
        """Load the model in a background thread; idempotent.

        Called after the handshake replies (``readyok``, or the first ``position``) rather than
        at start-up: loading the inference libraries holds the interpreter lock while the
        operating system checks each library, which would delay ``uciok`` and ``readyok`` on a
        first launch. Errors are left for the first command that needs the model, which loads
        again and reports them on the normal path.
        """
        if self._preload is not None or self.dsc is not None:
            return

        def preload() -> None:
            try:
                self.ensure_loaded()
            except Exception:  # noqa: BLE001 - reported by the foreground load
                return

        self._preload = threading.Thread(target=preload, name="dsc-preload", daemon=True)
        self._preload.start()

    def _name(self) -> str:
        if self.dsc is not None:
            model_id = self.dsc.model_id
        elif self.args.model is not None and self._registered(self.args.model):
            # A registered model (the standalone engines pass both --model and the local
            # weights) is named without opening the weights, which would import torch.
            model_id = self.args.model
        elif self.args.checkpoint_path:
            path = Path(self.args.checkpoint_path)
            if path.is_dir():
                found = sorted(path.glob("*.safetensors"))
                path = found[0] if len(found) == 1 else path
            try:
                model_id = read_config(path)["model_id"]
            except (OSError, ValueError, KeyError):
                model_id = path.stem
        else:
            model_id = self.args.model
        # Checkpoint identities retain their version; the GUI uses the public name.
        for entry in MODELS:
            if model_id == entry.weights.removesuffix(".safetensors"):
                return entry.repo_name
        try:
            return resolve(model_id).repo_name
        except KeyError:
            return f"{ENGINE_NAME} {model_id}"

    @staticmethod
    def _registered(model_id: str) -> bool:
        try:
            resolve(model_id)
        except KeyError:
            return False
        return True

    # ── prediction ──────────────────────────────────────────────────────────

    def _input(self, board: chess.Board) -> PositionInput:
        from .predictor import PositionInput

        white_elo, black_elo = (
            (self.self_elo, self.oppo_elo)
            if self.board.turn == chess.WHITE
            else (self.oppo_elo, self.self_elo)
        )
        return PositionInput(
            board,
            white_elo=white_elo,
            black_elo=black_elo,
            time_control=self.time_control,
            rating_system=self.rating_system,
            use_history=None if self.args.use_uci_history else False,
        )

    def _predict(self, items: list[PositionInput]) -> list[Prediction]:
        self.forward_calls += 1
        self.forward_positions += len(items)
        return self.ensure_loaded().predict_batch(items)

    def analyse(self, allowed: set[str] | None = None) -> Analysis | None:
        """The move to play and the candidates to show, or None when there is no move to give
        (a finished game, or a restriction that leaves no legal move: python-chess sends
        ``searchmoves 0000`` for an empty one), which answers ``bestmove 0000`` without info
        lines.

        The shown candidates keep their unrestricted probabilities. ``allowed`` (from ``go
        searchmoves``) limits them to those moves, and the move played is drawn from the allowed
        moves rescaled to sum to one. The candidates' auxiliary WDL is computed only in the modes
        that show it.
        """
        if self.dsc is None and self._preload is not None and self._preload.is_alive():
            self.log(
                f"{self._name()}: waiting for the model to finish loading (the first launch "
                "on a computer can take 10-20 seconds)"
            )
        self.ensure_loaded()
        from .sampling import select_move

        if self.board.is_game_over() or not any(self.board.legal_moves):
            return None
        pred = self._predict([self._input(self.board)])[0]
        candidates = pred.moves
        analysis = Analysis(chosen="", prediction=pred, shown=[])
        if allowed is not None:
            candidates = {u: p for u, p in pred.moves.items() if u in allowed}
            if not candidates:
                return None
            total = sum(candidates.values())
            analysis.allowed = allowed
            analysis.allowed_mass = total
            analysis.n_allowed = len(candidates)
            selection = (
                {u: p / total for u, p in candidates.items()}
                if total > 0
                else {u: 1.0 / len(candidates) for u in candidates}
            )
        else:
            selection = candidates
        analysis.chosen = select_move(
            selection, temperature=self.temperature, top_p=self.top_p, rng=self.rng
        )
        analysis.shown = list(candidates.items())[: self.multipv]
        if self.output_mode in CANDIDATE_WDL_MODES:
            # WDL of each candidate: evaluate the position after the move (opponent to move)
            # and turn its [win, draw, loss] around to the mover's point of view.
            children = []
            for uci, _ in analysis.shown:
                child = self.board.copy()
                child.push_uci(uci)
                children.append(self._input(child))
            for (uci, _), cp in zip(analysis.shown, self._predict(children), strict=True):
                analysis.wdl[uci] = (cp.wdl.loss, cp.wdl.draw, cp.wdl.win)
        return analysis

    def _wdl_line(self, rank: int, uci: str, wdl) -> str:
        win, draw, loss = permille(wdl)
        return f"info depth 1 multipv {rank} score cp {win - loss} wdl {win} {draw} {loss} pv {uci}"

    def _context(self, a: Analysis) -> str:
        i = a.prediction.inputs
        white, black = (
            (i.elo_mover, i.elo_opponent)
            if self.board.turn == chess.WHITE
            else (i.elo_opponent, i.elo_mover)
        )
        if i.repetition_source == "assumed_none":
            repetition = "repetition unknown (no complete move list), treated as none"
        else:
            seen = {
                (False, False): "not seen before",
                (True, False): "seen once before",
                (True, True): "seen twice before",
            }
            source = "from the move list" if i.repetition_source == "move_stack" else "as given"
            repetition = f"position {seen[tuple(i.repetition)]} ({source})"
        return (
            f"{self._name()}: P = probability that the player to move plays the move "
            f"(a prediction of human choice, not an evaluation); White {white:g}, Black "
            f"{black:g}, {i.time_control}, {i.rating_system} ratings; {i.history_boards} "
            f"previous positions; {repetition}; move played: temperature {self.temperature:g}, "
            f"top-p {self.top_p:g}"
        )

    def _summary(self, a: Analysis) -> str:
        n_legal = len(a.prediction.moves)
        ranked = ", ".join(
            f"{self.board.san(chess.Move.from_uci(u))} {percent(p)}%" for u, p in a.shown
        )
        shown_mass = sum(p for _, p in a.shown)
        text = (
            f"move probabilities, highest first: {ranked} ({len(a.shown)} of {n_legal} legal "
            f"moves shown, together {percent(shown_mass)}%)"
        )
        if a.allowed is not None:
            text += (
                f"; searchmoves allows {a.n_allowed} of {n_legal} moves, together "
                f"{percent(a.allowed_mass)}%: their probabilities are shown unrestricted, and the "
                f"move played is drawn from them rescaled to 100%"
            )
        return text

    def _score_note(self) -> str:
        if self.output_mode != "probability-as-score":
            return ""
        return (
            f"; score = probability x {SCORE_PER_PROBABILITY} (+41.73 means 41.73%), signed so "
            f"that GUIs showing scores from White's side print it positive (sent negative with "
            f"Black to move); not an evaluation"
        )

    def output_lines(self, a: Analysis) -> list[str]:
        """The ``info`` lines of one ``go`` in the current output mode."""
        lines = [f"info string DSC {self._context(a)}{self._score_note()}"]
        # Lc0's verbose move statistics form, least probable first: GUIs that read it (Nibbler)
        # rank the moves by arrival, the last one best.
        lines += [f"info string {uci} (P: {percent(p)}%)" for uci, p in reversed(a.shown)]
        for rank, (uci, p) in enumerate(a.shown, start=1):
            if self.output_mode == "probability-as-score":
                # Negative with Black to move: White's-point-of-view GUIs then print it positive.
                sign = 1 if self.board.turn == chess.WHITE else -1
                score = sign * int(round(p * SCORE_PER_PROBABILITY))
                lines.append(f"info depth 1 multipv {rank} score cp {score} pv {uci}")
            elif uci in a.wdl:
                lines.append(self._wdl_line(rank, uci, a.wdl[uci]))
            else:
                lines.append(f"info depth 1 multipv {rank} pv {uci}")
        lines.append(f"info string DSC {self._summary(a)}")
        return lines

    # ── protocol ────────────────────────────────────────────────────────────

    def cmd_uci(self) -> None:
        self.send(f"id name {self._name()}")
        self.send(f"id author {ENGINE_AUTHOR}")
        elo = int(round(self.args.elo))
        self.send(f"option name Elo type spin default {elo} min 0 max 5000")
        self.send(f"option name SelfElo type spin default {elo} min 0 max 5000")
        self.send(f"option name OppoElo type spin default {elo} min 0 max 5000")
        self.send(f"option name Temperature type string default {self.args.temperature}")
        self.send(f"option name TopP type string default {self.args.top_p}")
        self.send(f"option name MultiPV type spin default {self.multipv} min 1 max {MAX_MULTIPV}")
        tcs = " ".join(f"var {t}" for t in TIME_CONTROLS)
        self.send(f"option name TimeControl type combo default {self.time_control} {tcs}")
        rss = " ".join(f"var {r}" for r in RATING_SYSTEMS)
        self.send(f"option name RatingSystem type combo default {self.rating_system} {rss}")
        modes = " ".join(f"var {m}" for m in OUTPUT_MODES)
        self.send(f"option name OutputMode type combo default {self.output_mode} {modes}")
        self.send("uciok")

    def cmd_setoption(self, line: str) -> None:
        try:
            after_name = line.split("name", 1)[1].strip()
            name, _, value = after_name.partition("value")
            name = name.strip().lower()
            value = value.strip()
        except (IndexError, ValueError):
            return
        try:
            if name == "elo":
                self.self_elo = self.oppo_elo = float(value)
            elif name == "selfelo":
                self.self_elo = float(value)
            elif name == "oppoelo":
                self.oppo_elo = float(value)
            elif name == "temperature":
                self.temperature = float(value)
            elif name == "topp":
                self.top_p = float(value)
            elif name == "multipv":
                self.multipv = self._clamp_multipv(value)
            elif name == "timecontrol" and value in TIME_CONTROLS:
                self.time_control = value
            elif name == "ratingsystem" and value in RATING_SYSTEMS:
                self.rating_system = value
            elif name == "outputmode" and value in OUTPUT_MODES:
                self.output_mode = value
        except ValueError:
            return

    def cmd_ucinewgame(self) -> None:
        self.board = chess.Board()
        self.pending_bestmove = None
        self.pending_search = False

    def cmd_position(self, line: str) -> None:
        tokens = line.split()
        if len(tokens) < 2:
            return
        i = 1
        if tokens[i] == "startpos":
            board = chess.Board()
            i += 1
        elif tokens[i] == "fen":
            fen_fields = []
            i += 1
            while i < len(tokens) and tokens[i] != "moves" and len(fen_fields) < 6:
                fen_fields.append(tokens[i])
                i += 1
            try:
                board = chess.Board(" ".join(fen_fields))
            except ValueError:
                return
        else:
            return
        if i < len(tokens) and tokens[i] == "moves":
            for uci in tokens[i + 1 :]:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    return
                if move not in board.legal_moves:
                    return
                board.push(move)
        self.board = board
        self.pending_bestmove = None
        self.pending_search = False

    def cmd_go(self, line: str) -> None:
        tokens = line.split()[1:]
        allowed: set[str] | None = None
        if "searchmoves" in tokens:
            allowed = set()
            i = tokens.index("searchmoves") + 1
            while i < len(tokens) and tokens[i] not in GO_KEYWORDS:
                if tokens[i] != "0000":  # python-chess encodes an empty restriction as 0000
                    allowed.add(tokens[i])
                i += 1
        analysis = self.analyse(allowed)
        chosen = analysis.chosen if analysis is not None else None
        for line in self.output_lines(analysis) if analysis is not None else []:
            self.send(line)
        # `go infinite` and `go ponder` hold the answer until `stop` / `ponderhit`. The engine
        # does not think on the opponent's time: the answer is already computed.
        if "infinite" in tokens or "ponder" in tokens:
            self.pending_bestmove = chosen
            self.pending_search = True
            return
        self.send_bestmove(chosen)

    def cmd_stop(self) -> None:
        if not self.pending_search:
            return
        self.send_bestmove(self.pending_bestmove)
        self.pending_bestmove = None
        self.pending_search = False

    def send_bestmove(self, move: str | None) -> None:
        self.send(f"bestmove {move or '0000'}")

    def run(self, stream=None) -> None:
        stream = stream or sys.stdin
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            head = line.split()[0]
            if line == "uci":
                self.cmd_uci()
            elif line == "isready":
                # Answered at once, as the protocol asks; the first go waits for the model.
                self.send("readyok")
                self.start_preload()
            elif line == "ucinewgame":
                self.start_preload()
                self.cmd_ucinewgame()
            elif head == "position":
                self.start_preload()
                self.cmd_position(line)
            elif head == "go":
                self.cmd_go(line)
            elif head == "setoption":
                self.cmd_setoption(line)
            elif line in ("stop", "ponderhit"):
                self.cmd_stop()
            elif line == "quit":
                return


def read_config(weights_path, config_path=None) -> dict:
    """The checkpoint's configuration (imports torch on first use; see ``checkpoint``)."""
    from .checkpoint import read_config as _read_config

    return _read_config(weights_path, config_path)


def __getattr__(name: str):
    # ``uci.DSC`` stays available for callers and tests without importing torch up front.
    if name == "DSC":
        from .predictor import DSC

        return DSC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv=None) -> None:
    args = parse_args(argv)
    Engine(args).run()


def _preset(model: str, argv) -> None:
    extra = list(sys.argv[1:] if argv is None else argv)
    main(["--temperature", "0", *extra, "--model", model])


def main_23m(argv=None) -> None:
    """``dsc-23m``: the 23M history model, most probable move, history from the move list."""
    _preset("dsc-23m", ["--use-uci-history", *(argv if argv is not None else sys.argv[1:])])


def main_23m_fen(argv=None) -> None:
    """``dsc-23m-fen``: the 23M FEN (no-history) model, most probable move."""
    _preset("dsc-23m-fen", argv)


def main_79m(argv=None) -> None:
    """``dsc-79m``: the 79M history model, most probable move, history from the move list."""
    _preset("dsc-79m", ["--use-uci-history", *(argv if argv is not None else sys.argv[1:])])


def main_79m_fen(argv=None) -> None:
    """``dsc-79m-fen``: the 79M FEN (no-history) model, most probable move."""
    _preset("dsc-79m-fen", argv)


def main_150m(argv=None) -> None:
    """``dsc-150m``: the 150M history model, most probable move, history from the move list."""
    _preset("dsc-150m", ["--use-uci-history", *(argv if argv is not None else sys.argv[1:])])


def main_150m_fen(argv=None) -> None:
    """``dsc-150m-fen``: the 150M FEN (no-history) model, most probable move."""
    _preset("dsc-150m-fen", argv)


if __name__ == "__main__":
    main()
