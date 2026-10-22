# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline UCI entrypoint for a relocatable, one-folder DSC distribution.

Only the small configuration is hashed here. The normal model loader verifies the
weights once, when UCI receives ``isready`` or ``go``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


class BundleError(ValueError):
    """The standalone distribution is incomplete or inconsistent."""


def _object(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise BundleError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{label} must contain a JSON object")
    return value


def _file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise BundleError(f"{relative} points outside the engine folder")
    if not path.is_file():
        raise BundleError(f"missing file: {relative}")
    return path


def _model_file(root: Path, value, field: str) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise BundleError(f"bundle.json has an invalid {field} path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "models":
        raise BundleError(f"bundle.json {field} must be a relative path inside models/")
    return _file(root, value)


def _text(metadata: dict, field: str) -> str:
    value = metadata.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BundleError(f"bundle.json lacks a valid {field}")
    return value


def resolve_arguments(root: str | Path, argv: list[str]) -> list[str]:
    """Validate bundle metadata and append settings that keep inference offline."""
    root = Path(root).resolve()
    metadata = _object(_file(root, "bundle.json").read_bytes(), "bundle.json")
    model_id = _text(metadata, "model_id")
    artifact_id = _text(metadata, "artifact_id")
    mode = metadata.get("mode")
    if mode not in ("history", "strict"):
        raise BundleError("bundle.json has an invalid model variant")
    for field in ("weights_sha256", "config_sha256"):
        if not isinstance(metadata.get(field), str) or not re.fullmatch(
            r"[0-9a-f]{64}", metadata[field]
        ):
            raise BundleError(f"bundle.json lacks a valid {field}")
    weights = _model_file(root, metadata.get("weights"), "weights")
    config = _model_file(root, metadata.get("config"), "config")
    if weights.suffix != ".safetensors" or config != weights.with_suffix(".json"):
        raise BundleError("the weights and JSON configuration must be adjacent with the same stem")
    raw_config = config.read_bytes()
    if hashlib.sha256(raw_config).hexdigest() != metadata["config_sha256"]:
        raise BundleError("the bundled JSON configuration failed its checksum")
    config_data = _object(raw_config, "the model configuration")
    expected = {
        "model_id": artifact_id,
        "mode": mode,
        "weights_sha256": metadata["weights_sha256"],
    }
    for field, value in expected.items():
        if config_data.get(field) != value:
            raise BundleError(f"the model configuration's {field} does not match bundle.json")
    # Put pinned settings last so a GUI cannot accidentally request another model,
    # GPU backend, or downloaded checkpoint. Temperature remains user-adjustable.
    args = [
        "--temperature",
        "0",
        *argv,
        "--device",
        "cpu",
        "--checkpoint-path",
        str(weights),
        "--model",
        model_id,
        "--local-files-only",
    ]
    if mode == "history":
        args.append("--use-uci-history")
    return args


def main(argv=None, bundle_root=None) -> int:
    """Run UCI using the adjacent bundle; report failures only on stderr."""
    if argv is None:
        argv = sys.argv[1:]
    if bundle_root is None:
        anchor = sys.executable if getattr(sys, "frozen", False) else __file__
        bundle_root = Path(anchor).resolve().parent
    try:
        args = resolve_arguments(bundle_root, list(argv))
        # Keep validation independent of Python packages and their native libraries.
        from chessds_dsc.uci import main as uci_main

        uci_main(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        detail = " ".join(str(exc).split()) or type(exc).__name__
        print(
            f"ChessDS DSC could not start or continue: {detail}. "
            "Keep the complete engine folder together. Re-extract the downloaded "
            "archive if files are missing or damaged.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
