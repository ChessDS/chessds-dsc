# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Public input vocabularies and defaults that need neither torch nor numpy.

The UCI engine answers ``uci`` from this module alone, before the inference libraries are
imported, so that chess GUIs see ``uciok`` within their start-up limits.
"""

from __future__ import annotations

# The values the public API accepts: every code that occurred in training. "correspondence"
# and the "unknown" codes have embedding rows but never occurred, so they are not offered.
TIME_CONTROLS = ("bullet", "blitz", "rapid", "classical")
RATING_SYSTEMS = ("fide", "lichess", "other")
VENUES = ("otb", "online")

DEFAULT_ELO = 2578.0
DEFAULT_TIME_CONTROL = "classical"
DEFAULT_RATING_SYSTEM = "fide"
