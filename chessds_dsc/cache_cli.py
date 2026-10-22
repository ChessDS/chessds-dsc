# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""``dsc-cache``: download registered models into the local cache ahead of time."""

from __future__ import annotations

import argparse
import sys

from .registry import MODELS, fetch, format_model_list, resolve


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="dsc-cache", description="Pre-download DSC models.")
    p.add_argument(
        "--model", action="append", dest="models", default=[], help="model id or alias (repeatable)"
    )
    p.add_argument("--all", action="store_true", help="download every registered model")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--list", action="store_true", help="list the registered models and exit")
    args = p.parse_args(argv)
    if args.list or (not args.models and not args.all):
        print(format_model_list())
        return
    entries = list(MODELS) if args.all else [resolve(m) for m in args.models]
    for entry in entries:
        weights, config = fetch(
            entry,
            cache_dir_=args.cache_dir,
            base_url_=args.base_url,
            force_download=args.force_download,
        )
        print(f"{entry.id}: {weights}", file=sys.stderr, flush=True)
