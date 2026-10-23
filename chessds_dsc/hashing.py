# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""File hashing shared by the registry and the checkpoint format (standard library only)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
