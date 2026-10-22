#!/usr/bin/env python3
# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify an unpacked standalone engine through UCI, without importing Torch.

    python standalone/verify.py --engine /path/to/dsc-23m --receipt check.json

Runs the executable without arguments, from an unrelated temporary directory containing spaces,
with fresh home/cache directories and no Python or ChessDS environment overrides. The standalone
entry point must itself select its bundled weights and forbid downloads. The verifier supplies an
unreachable download URL as an additional guard. Requires the lightweight `chess` package for
legal-move checks. Existing receipts are never overwritten. Exit 0 means every check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

MAX_OUTPUT = 64 * 1024
MOVES = "e2e4 e7e5 g1f3"
COMMANDS = (
    "uci\nisready\nucinewgame\n"
    f"position startpos moves {MOVES}\n"
    "setoption name OutputMode value likelihood-wdl\ngo\n"
    "setoption name OutputMode value likelihood\ngo\n"
    "setoption name OutputMode value probability-as-score\ngo\n"
    "setoption name OutputMode value likelihood\ngo infinite\nisready\nstop\n"
    "position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\ngo\nquit\n"
)
P_LINE = re.compile(r"^info string (\S+) \(P: ([0-9.]+)%\)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isolated_environment(work: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
        and not key.upper().startswith("CHESSDS_")
    }
    home, cache = work / "fresh home", work / "fresh cache"
    home.mkdir()
    cache.mkdir()
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CACHE_HOME=str(cache),
        LOCALAPPDATA=str(cache),
        APPDATA=str(cache),
        CHESSDS_DSC_CACHE=str(cache / "models"),
        CHESSDS_DSC_BASE_URL="http://127.0.0.1:9",
    )
    return env


def run_session(engine: Path, cwd: Path, env: dict, timeout: float) -> dict:
    started = time.monotonic()
    result = {
        "returncode": None,
        "timed_out": False,
        "startup_seconds": None,
        "ready_seconds": None,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    process = subprocess.Popen(
        [str(engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
    )

    def drain(stream, name):
        kept, size = [], 0
        try:
            for chunk in iter(lambda: stream.readline(8192), ""):
                if name == "stdout":
                    stamp = round(time.monotonic() - started, 6)
                    if chunk.strip() == "uciok" and result["startup_seconds"] is None:
                        result["startup_seconds"] = stamp
                    if chunk.strip() == "readyok" and result["ready_seconds"] is None:
                        result["ready_seconds"] = stamp
                room = MAX_OUTPUT - size
                if room > 0:
                    kept.append(chunk[:room])
                    size += min(len(chunk), room)
                if len(chunk) > room:
                    result[f"{name}_truncated"] = True
        finally:
            result[name] = "".join(kept)

    readers = [
        threading.Thread(target=drain, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        try:
            process.stdin.write(COMMANDS)
            process.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            result["stdin_error"] = str(exc)
        try:
            process.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            process.kill()
            process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for reader in readers:
            reader.join(timeout=1)
        result["streams_closed"] = not any(reader.is_alive() for reader in readers)
        result["returncode"] = process.returncode
        result["session_seconds"] = round(time.monotonic() - started, 6)
        if result["streams_closed"]:
            process.stdout.close()
            process.stderr.close()
    return result


def protocol_checks(stdout: str) -> dict:
    import chess

    board = chess.Board()
    for move in MOVES.split():
        board.push_uci(move)
    legal = {move.uci() for move in board.legal_moves}
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    best, sections, current, ready_counts, unknown = [], [], [], [], []
    for line in lines:
        if line == "readyok":
            ready_counts.append(len(best))
        elif line.startswith("bestmove "):
            tokens = line.split()
            best.append(tokens[1] if len(tokens) == 2 else "<malformed>")
            sections.append(current)
            current = []
        elif line.startswith("info "):
            current.append(line)
        elif line != "uciok" and not line.startswith(("id name ", "id author ", "option name ")):
            unknown.append(line)
    modes = []
    for index, mode in enumerate(
        ("likelihood-wdl", "likelihood", "probability-as-score", "likelihood")
    ):
        section = sections[index] if index < len(sections) else []
        p_lines = [P_LINE.fullmatch(line) for line in section if P_LINE.fullmatch(line)]
        rows = [line for line in section if line.startswith("info depth ")]
        has_score = any(" score cp " in line for line in rows)
        has_wdl = any(" wdl " in line for line in rows)
        expected_score, expected_wdl = mode != "likelihood", mode == "likelihood-wdl"
        probabilities_ok = bool(p_lines) and all(
            match[1] in legal and 0 <= float(match[2]) <= 100 for match in p_lines
        )
        pv_ok = bool(rows) and all(
            " pv " in line and line.split(" pv ", 1)[1].split()[0] in legal for line in rows
        )
        modes.append(
            {
                "mode": mode,
                "pass": probabilities_ok
                and pv_ok
                and has_score == expected_score
                and has_wdl == expected_wdl,
            }
        )
    handshake = (
        lines.count("uciok") == 1
        and bool(ready_counts)
        and lines.index("uciok") < lines.index("readyok")
        and any(line.startswith("id name ") for line in lines)
        and any(line.startswith("option name OutputMode ") for line in lines)
    )
    checks = {
        "handshake": handshake,
        "legal_bestmoves": len(best) == 5 and all(move in legal for move in best[:4]),
        "terminal_bestmove": len(best) == 5 and best[-1] == "0000",
        "infinite_waits_for_stop": ready_counts == [0, 3],
        "output_modes": all(mode["pass"] for mode in modes),
        "protocol_only_stdout": not unknown,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "bestmoves": best,
        "ready_after_bestmove_counts": ready_counts,
        "modes": modes,
        "unexpected_stdout": unknown[:10],
    }


def verify(engine: Path, receipt: Path, timeout: float = 90) -> int:
    # Reserve before launching so failures are recorded and no prior receipt can be replaced.
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination = receipt.open("x")
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite receipt {receipt}") from None
    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 (Python 3.10)
        "machine": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "engine": str(engine.resolve()),
        "timeout_seconds": timeout,
        "pass": False,
        "returncode": None,
        "startup_seconds": None,
        "ready_seconds": None,
        "session_seconds": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }
    try:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        result["executable_sha256"] = sha256_file(engine)
        with tempfile.TemporaryDirectory(prefix="dsc standalone check ") as tmp:
            work = Path(tmp)
            session = run_session(engine.resolve(), work, isolated_environment(work), timeout)
        result.update(session)
        result["protocol"] = protocol_checks(session["stdout"])
        result["pass"] = (
            session["returncode"] == 0
            and not session["timed_out"]
            and session["streams_closed"]
            and not session["stdout_truncated"]
            and not session.get("stdin_error")
            and result["protocol"]["pass"]
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with destination:
            json.dump(result, destination, indent=2)
            destination.write("\n")
    print(f"{'PASS' if result['pass'] else 'FAIL'}: {engine}; receipt {receipt}")
    return 0 if result["pass"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    raise SystemExit(verify(args.engine, args.receipt, args.timeout))


if __name__ == "__main__":
    main()
