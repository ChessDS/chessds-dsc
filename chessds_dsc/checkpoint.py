# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""The DSC weight file format: safetensors weights plus a JSON configuration.

A model is two files with the same stem, ``<name>.safetensors`` and ``<name>.json``. The JSON
holds everything needed to build and use the model:

    {
      "format": "chessds-dsc-v1",
      "model_id": "dsc-23m-v1",
      "mode": "history" | "strict",
      "architecture": {... DSCConfig fields ...},
      "parameters": 22538566,
      "encoding_version": "dsc-encoding-v1",
      "defaults": {"elo": 2578, "time_control": "classical", "rating_system": "fide"},
      "weights_sha256": "...",
      ...
    }

The safetensors header repeats ``format``, ``model_id``, ``mode`` and ``architecture`` (as a JSON
string) in its metadata, so a weights file alone is still loadable. Loading never unpickles.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .encoding import ENCODING_VERSION
from .model import DSCConfig, DSCModel, count_parameters

FORMAT = "chessds-dsc-v1"
MODES = ("history", "strict")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_path_for(weights_path: str | Path) -> Path:
    return Path(weights_path).with_suffix(".json")


def save_model(
    model: DSCModel,
    stem: str | Path,
    *,
    model_id: str,
    mode: str,
    defaults: dict,
    extra: dict | None = None,
) -> dict:
    """Write ``<stem>.safetensors`` and ``<stem>.json``; return the written configuration."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    weights_path = stem.with_suffix(".safetensors")
    arch = model.cfg.to_dict()
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    _check_defaults(defaults)
    n_params = count_parameters(model)
    # The complete input contract travels inside the weights file too.
    metadata = {
        "format": FORMAT,
        "model_id": model_id,
        "mode": mode,
        "architecture": json.dumps(arch, sort_keys=True),
        "encoding_version": ENCODING_VERSION,
        "defaults": json.dumps(dict(defaults), sort_keys=True),
        "parameters": str(n_params),
    }
    save_file(state, str(weights_path), metadata=metadata)
    config = {
        "format": FORMAT,
        "model_id": model_id,
        "mode": mode,
        "architecture": arch,
        "parameters": n_params,
        "encoding_version": ENCODING_VERSION,
        "defaults": dict(defaults),
        "weights_file": weights_path.name,
        "weights_sha256": sha256_file(weights_path),
    }
    if extra:
        config.update(extra)
    with open(config_path_for(weights_path), "w") as f:
        json.dump(config, f, indent=1, sort_keys=True)
        f.write("\n")
    return config


def read_config(weights_path: str | Path, config_path: str | Path | None = None) -> dict:
    """The configuration of a weights file: its JSON sidecar when present, else the metadata
    embedded in the safetensors header."""
    weights_path = Path(weights_path)
    config_path = Path(config_path) if config_path else config_path_for(weights_path)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt") as f:
            meta = f.metadata() or {}
        needed = ("format", "model_id", "mode", "architecture", "encoding_version", "defaults")
        missing = [k for k in needed if k not in meta]
        if missing:
            raise ValueError(
                f"{weights_path} has no usable configuration: no {config_path.name} next to it "
                f"and its metadata lacks {missing}"
            )
        config = {k: meta[k] for k in needed}
        config["architecture"] = json.loads(meta["architecture"])
        config["defaults"] = json.loads(meta["defaults"])
        if "parameters" in meta:
            config["parameters"] = int(meta["parameters"])
    if config.get("format") != FORMAT:
        raise ValueError(f"unsupported weights format {config.get('format')!r}; expected {FORMAT}")
    if config.get("encoding_version") != ENCODING_VERSION:
        raise ValueError(
            f"unsupported encoding version {config.get('encoding_version')!r}; this package "
            f"implements {ENCODING_VERSION}"
        )
    if config.get("mode") not in MODES:
        raise ValueError(f"unsupported mode {config.get('mode')!r}")
    _check_defaults(config.get("defaults"))
    return config


DEFAULT_KEYS = ("elo", "time_control", "rating_system")


def _check_defaults(defaults) -> None:
    missing = [k for k in DEFAULT_KEYS if not isinstance(defaults, dict) or k not in defaults]
    if missing:
        raise ValueError(f"the configuration's input defaults lack {missing}")


def load_model(
    weights_path: str | Path,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    *,
    verify: bool = True,
) -> tuple[DSCModel, dict]:
    """Build a :class:`DSCModel` from a weights file and return it in eval mode with its config.

    With ``verify`` the file's sha256 is checked against ``weights_sha256`` in the configuration
    when that field is present.
    """
    weights_path = Path(weights_path)
    config = read_config(weights_path, config_path)
    expected = config.get("weights_sha256")
    if verify and expected:
        actual = sha256_file(weights_path)
        if actual != expected:
            raise ValueError(
                f"{weights_path} is corrupt or does not match its configuration: "
                f"sha256 {actual} != {expected}"
            )
    cfg = DSCConfig.from_dict(config["architecture"])
    model = DSCModel(cfg)
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config
