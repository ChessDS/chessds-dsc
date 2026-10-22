# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""The published models, and downloading them into a local cache.

The download base URL comes from the ``CHESSDS_DSC_BASE_URL`` environment variable or the ``base_url``
argument. Files are expected at ``<base>/<repo>/resolve/<revision>/<file name>``: the layout of
a Hugging Face repository per model (base ``https://huggingface.co/<organisation>``), which a
plain HTTP or S3 mirror reproduces with the same paths, so switching hosts changes only the
base. ``CHESSDS_DSC_URL_TEMPLATE`` can change the layout (placeholders ``{base}``, ``{id}``,
``{repo}``, ``{revision}``, ``{file}``). File names carry the artifact version, and downloads
are verified against the sha256 recorded here before they are used, and cached under
``$CHESSDS_DSC_CACHE`` (default ``~/.cache/chessds-dsc``).

Only the models listed here can be downloaded by name; larger private models are not part of
this registry. Any weights file in the DSC format can still be loaded from a local path.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import sha256_file

ENV_BASE_URL = "CHESSDS_DSC_BASE_URL"
ENV_CACHE = "CHESSDS_DSC_CACHE"
ENV_URL_TEMPLATE = "CHESSDS_DSC_URL_TEMPLATE"
DEFAULT_BASE_URL: str | None = "https://huggingface.co/ChessDS"
# How a file's URL is built. "{base}/{repo}/resolve/{revision}/{file}" is a Hugging Face
# repository per model (base "https://huggingface.co/<organisation>") and, with another base,
# a mirror that copies the same paths.
DEFAULT_URL_TEMPLATE = "{base}/{repo}/resolve/{revision}/{file}"
REGISTRY_VERSION = "2026-10-20"  # bump whenever an entry is added or its files change


@dataclass(frozen=True)
class ModelEntry:
    id: str
    aliases: tuple[str, ...]
    size: str  # "23m", "79m", "150m"
    mode: str  # "history" or "strict"
    parameters: int
    weights: str  # file name, unique across models; carries the artifact version
    config: str  # file name, unique across models
    weights_sha256: str
    config_sha256: str
    description: str
    repo: str = ""  # hosting repository name; empty = the id
    revision: str = "v1"  # hosting revision (a tag) the files are fetched from

    @property
    def files(self) -> tuple[tuple[str, str], ...]:
        return ((self.weights, self.weights_sha256), (self.config, self.config_sha256))

    @property
    def repo_name(self) -> str:
        return self.repo or self.id


# Hashes are those of the released files (docs/dsc_release/release_v1/manifest.json). Every model
# is the completed fine-tune (step 17,454) of an extended training run. The public id is the
# model's name; the -h / -s / -strict aliases are the pre-release ids. Files are fetched from
# https://huggingface.co/ChessDS/<repo>/resolve/v1/.
MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        id="dsc-23m",
        aliases=("dsc-23m-h", "23m", "23m-h", "dsc-23m-history"),
        size="23m",
        mode="history",
        parameters=22_538_566,
        weights="dsc-23m-v1.safetensors",
        config="dsc-23m-v1.json",
        weights_sha256="740d9393d041508b872643b0393569fdbb1662bb8a19e24159d980343d9f3189",
        config_sha256="26d70ab23179cc3ec5f46d6511166015287161c7802d0ad686ab67678ac73328",
        description="23M, uses the previous 7 positions",
        repo="DSC-23M",
    ),
    ModelEntry(
        id="dsc-23m-fen",
        aliases=("dsc-23m-s", "23m-fen", "23m-s", "dsc-23m-strict", "23m-strict"),
        size="23m",
        mode="strict",
        parameters=22_538_566,
        weights="dsc-23m-fen-v1.safetensors",
        config="dsc-23m-fen-v1.json",
        weights_sha256="6caf893e9b960ae0916ecf05b3ff356c9dbf6736712ba77647152363ef8ba240",
        config_sha256="30bf19bdeeff1014810d1452b274953f384ea219dd4b5987f9e2c2b4db4b872b",
        description="23M FEN: no previous positions; for positions without a game record",
        repo="DSC-23M-FEN",
    ),
    ModelEntry(
        id="dsc-79m",
        aliases=("dsc-79m-h", "79m", "79m-h", "dsc-79m-history"),
        size="79m",
        mode="history",
        parameters=77_053_254,
        weights="dsc-79m-v1.safetensors",
        config="dsc-79m-v1.json",
        weights_sha256="883d7ec3b90c210ad9e2aa1b415db117955c8dded3e63ec45e464e8f89dfd233",
        config_sha256="79a34c15dbc3cea6b3ca3dcca4a84a041d0cf2d6a80593900a76dde6fb883f6f",
        description="79M, uses the previous 7 positions",
        repo="DSC-79M",
    ),
    ModelEntry(
        id="dsc-79m-fen",
        aliases=("dsc-79m-s", "79m-fen", "79m-s", "dsc-79m-strict", "79m-strict"),
        size="79m",
        mode="strict",
        parameters=77_053_254,
        weights="dsc-79m-fen-v1.safetensors",
        config="dsc-79m-fen-v1.json",
        weights_sha256="0851960d5c35ee0e4fbfc09499e304fad6689c8102fbc70a7d9dcc91451f00da",
        config_sha256="e88c86575ec983395bc07315eaefb6373207f87274348304a43bba07692d3c11",
        description="79M FEN: no previous positions",
        repo="DSC-79M-FEN",
    ),
    ModelEntry(
        id="dsc-150m",
        aliases=("dsc-150m-h", "150m", "150m-h", "dsc-150m-history"),
        size="150m",
        mode="history",
        parameters=150_907_206,
        weights="dsc-150m-v1.safetensors",
        config="dsc-150m-v1.json",
        weights_sha256="66b21d7882bb3fb9737bc798e709671325838e5469ed6bfcf114b3abbcb48e24",
        config_sha256="6916d0ed2b33a4f41c71e704bd23f3ebd6676759d70d5e3b009a6a39d3315e42",
        description="150M, uses the previous 7 positions; the most accurate model",
        repo="DSC-150M",
    ),
    ModelEntry(
        id="dsc-150m-fen",
        aliases=("dsc-150m-s", "150m-fen", "150m-s", "dsc-150m-strict", "150m-strict"),
        size="150m",
        mode="strict",
        parameters=150_907_206,
        weights="dsc-150m-fen-v1.safetensors",
        config="dsc-150m-fen-v1.json",
        weights_sha256="6392b549d635b85b2a7615f5fffeadf29aebc67f6da869352b4411ff558c19d3",
        config_sha256="c0d3462dbd6ade150ed81c9556329ce8218050eb2fe06c34b4a9062b8d3f1944",
        description="150M FEN: no previous positions; the most accurate FEN model",
        repo="DSC-150M-FEN",
    ),
)


def resolve(name: str) -> ModelEntry:
    """The registry entry for a model id or alias (case-insensitive)."""
    key = name.strip().lower()
    for entry in MODELS:
        if key == entry.id or key in entry.aliases:
            return entry
    known = ", ".join(e.id for e in MODELS) or "(none)"
    raise KeyError(f"unknown model {name!r}; known models: {known}")


def format_model_list() -> str:
    if not MODELS:
        return "no models are registered in this build"
    rows = [
        f"registry {REGISTRY_VERSION}",
        f"{'id':<16} {'size':<6} {'variant':<8} {'params':>12}  description",
    ]
    for e in MODELS:
        variant = "FEN" if e.mode == "strict" else "main"
        rows.append(f"{e.id:<16} {e.size:<6} {variant:<8} {e.parameters:>12,}  {e.description}")
    return "\n".join(rows)


def cache_dir(override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    env = os.environ.get(ENV_CACHE)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "chessds-dsc"


def base_url(override: str | None = None) -> str | None:
    url = override or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL
    return url.rstrip("/") if url else None


def file_url(entry: ModelEntry, name: str, base_url_: str | None = None) -> str:
    """The download URL of one of the entry's files, or a ``RuntimeError`` when no location is
    configured."""
    template = os.environ.get(ENV_URL_TEMPLATE) or DEFAULT_URL_TEMPLATE
    base = base_url(base_url_)
    if "{base}" in template and not base:
        raise RuntimeError(
            f"no download location is configured for {entry.id}: set {ENV_BASE_URL} or pass "
            "base_url, or load a local weights file with checkpoint_path"
        )
    return template.format(
        base=base or "", id=entry.id, repo=entry.repo_name, revision=entry.revision, file=name
    )


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as resp:
            shutil.copyfileobj(resp, out, length=1 << 20)
        actual = sha256_file(tmp)
        if actual != expected_sha256:
            raise ValueError(
                f"downloaded {url} has sha256 {actual}, expected {expected_sha256}; not installed"
            )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch(
    entry: ModelEntry,
    *,
    cache_dir_: str | Path | None = None,
    base_url_: str | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
    log=None,
) -> tuple[Path, Path]:
    """Ensure the entry's files are in the cache and return ``(weights_path, config_path)``.

    A cached file whose hash does not match the registry is re-downloaded (or rejected with
    ``local_files_only``).
    """
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    directory = cache_dir(cache_dir_) / entry.id
    paths = []
    for name, sha in entry.files:
        dest = directory / name
        ok = dest.exists() and not force_download and sha256_file(dest) == sha
        if not ok:
            if dest.exists() and not force_download:
                log(f"cached {dest} does not match the registry; re-downloading")
            if local_files_only:
                raise FileNotFoundError(
                    f"{dest} is not in the cache (or is corrupt) and local_files_only is set"
                )
            url = file_url(entry, name, base_url_)
            log(f"downloading {url}")
            try:
                _download(url, dest, sha)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"could not download {url}: {exc}") from exc
        paths.append(dest)
    return paths[0], paths[1]
