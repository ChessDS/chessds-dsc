#!/usr/bin/env python3
# Copyright 2026 Albert W. Hamood (ChessDS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build an offline UCI folder and archive for the current host, without downloading anything.

Invoke with the Python environment to freeze (CPU PyTorch on Linux/Windows)::

    python standalone/build.py --weights /absolute/model.safetensors \
        --weights-license /absolute/LICENSE-weights --output /new/output

--output must not exist. A first build leaves runtime/ (DSC/ plus runtime_receipt.json)
there; --runtime-dir can reuse that cache after source, environment and file checks.
This creates local, unsigned/unnotarized release candidates; it never publishes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 2 * 1024**3
SCHEMA_VERSION = 1
CHUNK = 1024 * 1024

# The child exits before freezing or copying weights; the builder never imports torch.
PROBE = r"""
import importlib.metadata as md
import json, pathlib, platform, re, sys, sysconfig
from dataclasses import asdict
import chess, torch
from chessds_dsc.registry import resolve
from chessds_dsc import __version__

deps = []
seen = set()
for d in md.distributions():
    name = d.metadata.get("Name") or "unknown"
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    if normalized in seen:
        continue
    seen.add(normalized)
    urls = d.metadata.get_all("Project-URL") or []
    homepage = d.metadata.get("Home-page")
    sources = [u.split(",", 1)[-1].strip() for u in urls
               if any(k in u.split(",", 1)[0].lower() for k in ("source", "code", "repository"))]
    if not sources and homepage:
        sources = [homepage]
    files = []
    for f in d.files or []:
        if any(p.lower().startswith(("license", "licence", "copying", "notice", "copyright"))
               for p in pathlib.PurePosixPath(str(f)).parts):
            path = pathlib.Path(d.locate_file(f))
            if path.is_file():
                files.append({"installed": str(path.resolve()), "relative": str(f)})
    deps.append({"name": name, "version": d.version, "source_urls": sources,
                 "project_urls": urls, "license": d.metadata.get("License-Expression") or d.metadata.get("License"),
                 "license_files": files})
candidates = [pathlib.Path(sys.base_prefix) / "LICENSE.txt", pathlib.Path(sys.base_prefix) / "LICENSE",
              pathlib.Path(sysconfig.get_path("stdlib")) / "LICENSE.txt"]
print(json.dumps({"entry": asdict(resolve(sys.argv[1])), "package_version": __version__,
    "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
               "executable": sys.executable, "source_url": "https://www.python.org/downloads/release/python-" + platform.python_version().replace(".", "") + "/",
               "license_candidates": [str(p) for p in candidates]},
    "torch": {"version": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip},
    "dependencies": sorted(deps, key=lambda d: (d["name"].lower(), d["version"])),
    "chess_source": str(pathlib.Path(chess.__file__).resolve().parent)}))
"""


class BuildError(RuntimeError):
    """A failed release precondition; existing output is never replaced."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_abs_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise BuildError(f"{label} must be the full path of an existing file: {path}")
    return path.resolve()


def refuse_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise BuildError(f"refusing to overwrite {path}")


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def copy_file(src: Path, dest: Path) -> None:
    """An exclusive, bounded-memory copy; metadata is retained after the bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as source, dest.open("xb") as target:
        shutil.copyfileobj(source, target, CHUNK)
    shutil.copystat(src, dest)


def host_tag(system: str | None = None, machine: str | None = None) -> str:
    system, machine = system or platform.system(), (machine or platform.machine()).lower()
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(machine)
    os_name = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(system)
    if not os_name or not arch or (system != "Darwin" and arch != "x64"):
        raise BuildError(f"unsupported standalone host: {system} {machine}")
    return f"{os_name}-{arch}"


def probe_environment(model: str, root: Path = PACKAGE_ROOT) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", PROBE, model], env=env, capture_output=True, text=True, timeout=120
    )
    if result.returncode:
        raise BuildError(f"environment probe failed: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise BuildError("environment probe did not return its JSON contract") from exc


def validate_cpu(probe: dict, system: str) -> None:
    if system in ("Linux", "Windows") and (probe["torch"].get("cuda") or probe["torch"].get("hip")):
        raise BuildError(
            "Linux/Windows standalone builds require a CPU-only PyTorch installation (no CUDA/ROCm)"
        )


def validate_artifacts(weights: Path, entry: dict) -> tuple[Path, dict]:
    """Verify exact registry artifacts before invoking PyInstaller."""
    config = weights.with_suffix(".json")
    if weights.name != entry["weights"] or config.name != entry["config"]:
        raise BuildError("weights/config filenames do not match the selected registry model")
    for path, expected in ((weights, entry["weights_sha256"]), (config, entry["config_sha256"])):
        if not path.is_file() or sha256(path) != expected:
            raise BuildError(f"artifact sha256 does not match registry: {path}")
    data = json.loads(config.read_text())
    if data.get("model_id") != weights.stem or data.get("mode") != entry["mode"]:
        raise BuildError("configuration identity or mode does not match the selected artifact")
    if (
        data.get("weights_sha256") != entry["weights_sha256"]
        or data.get("weights_file") != weights.name
    ):
        raise BuildError("configuration does not identify the verified weights")
    return config, data


def source_hashes(root: Path = PACKAGE_ROOT) -> dict:
    paths = [
        root / "standalone" / "engine.py",
        root / "standalone" / "build.py",
        root / "pyproject.toml",
    ]
    paths += sorted((root / "chessds_dsc").rglob("*.py"))
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def runtime_identity(probe: dict, tag: str, root: Path = PACKAGE_ROOT) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": tag,
        "python": {k: probe["python"][k] for k in ("version", "implementation")},
        "dependencies": {d["name"]: d["version"] for d in probe["dependencies"]},
        "torch": probe["torch"],
        "source_sha256": source_hashes(root),
        "freezer": "PyInstaller --onedir --console --name DSC",
    }


def inventory(directory: Path, *, allow_symlinks: bool = True) -> dict:
    out = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            if not allow_symlinks:
                raise BuildError(f"Windows archives cannot contain POSIX symlinks: {relative}")
            if not path.resolve().is_relative_to(directory.resolve()) or not path.exists():
                raise BuildError(f"runtime symlink is broken or leaves its folder: {relative}")
            out[relative] = {"symlink": os.readlink(path), "mode": mode}
        elif path.is_file():
            out[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path), "mode": mode}
    return out


def validate_runtime(directory: Path, identity: dict) -> Path:
    receipt = directory / "runtime_receipt.json"
    runtime = directory / "DSC"
    if not receipt.is_file() or not runtime.is_dir():
        raise BuildError("runtime reuse requires runtime_receipt.json and the DSC directory")
    record = json.loads(receipt.read_text())
    if record.get("identity") != identity:
        raise BuildError("runtime reuse refused: source or environment identity mismatch")
    actual = inventory(runtime, allow_symlinks=not identity["platform"].startswith("windows"))
    if actual != record.get("files"):
        raise BuildError("runtime reuse refused: runtime file hashes or symlinks changed")
    return runtime


def freeze_runtime(directory: Path, identity: dict, root: Path = PACKAGE_ROOT) -> Path:
    refuse_existing(directory)
    directory.mkdir()
    with tempfile.TemporaryDirectory(prefix="dsc-pyinstaller-") as scratch:
        temp = Path(scratch)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["CUDA_VISIBLE_DEVICES"] = ""
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onedir",
            "--console",
            "--name",
            "DSC",
            "--paths",
            str(root),
            "--distpath",
            str(directory),
            "--workpath",
            str(temp / "work"),
            "--specpath",
            str(temp),
            "--clean",
            str(root / "standalone" / "engine.py"),
        ]
        for unused in ("pytest", "IPython", "matplotlib", "torchvision", "torchaudio", "tkinter"):
            command[3:3] = ["--exclude-module", unused]
        subprocess.run(command, env=env, check=True)
    runtime = directory / "DSC"
    executable = runtime / ("DSC.exe" if identity["platform"].startswith("windows") else "DSC")
    if not executable.is_file() or not (runtime / "_internal").is_dir():
        raise BuildError("PyInstaller did not produce the expected onedir runtime")
    files = inventory(runtime, allow_symlinks=not identity["platform"].startswith("windows"))
    write_json(
        directory / "runtime_receipt.json",
        {"identity": identity, "files": files, "command": command},
    )
    return runtime


def source_snapshot(bundle: Path, probe: dict, root: Path = PACKAGE_ROOT) -> None:
    source = bundle / "source"
    # Include the package's editable source, examples, tests and build recipe; never caches/dist/weights.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in {"dist", "build", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
            or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_file() and (
            path.suffix in {".py", ".toml", ".md", ".txt"}
            or path.name in {"LICENSE", "MANIFEST.in"}
        ):
            copy_file(path, source / "chessds-dsc" / relative)
    chess = Path(probe["chess_source"])
    for path in sorted(chess.rglob("*.py")):
        copy_file(path, source / "third-party" / "chess" / "chess" / path.relative_to(chess))
    chess_dist = next((d for d in probe["dependencies"] if d["name"].lower() == "chess"), None)
    if not chess_dist or not chess_dist["license_files"]:
        raise BuildError("the installed chess distribution must provide its GPL licence and source")
    for i, item in enumerate(chess_dist["license_files"]):
        name = "LICENSE" if i == 0 else f"LICENSE-{i}"
        copy_file(Path(item["installed"]), source / "third-party" / "chess" / name)
    write_json(
        source / "third-party" / "chess" / "provenance.json",
        {k: chess_dist[k] for k in ("name", "version", "source_urls")},
    )


def collect_licenses(bundle: Path, probe: dict, python_license: Path | None) -> None:
    licenses = bundle / "third_party_licenses"
    licenses.mkdir()
    for dist in probe["dependencies"]:
        name = dist["name"].replace("/", "_").replace("\\", "_")
        for item in dist["license_files"]:
            parts = [
                "_up_" if p == ".." else p
                for p in Path(item["relative"]).parts
                if p not in {"/", "."}
            ]
            copy_file(Path(item["installed"]), licenses / name / Path(*parts))
    candidates = (
        [python_license]
        if python_license
        else [Path(p) for p in probe["python"]["license_candidates"]]
    )
    found = next((p for p in candidates if p and p.is_file()), None)
    if found is None:
        raise BuildError("Python licence not found; pass --python-license with its full local path")
    copy_file(found, licenses / "Python" / "LICENSE.txt")
    metadata = [
        {k: d[k] for k in ("name", "version", "source_urls", "project_urls", "license")}
        for d in probe["dependencies"]
    ]
    write_json(
        licenses / "versions_and_sources.json",
        {
            "python": {k: probe["python"][k] for k in ("version", "implementation", "source_url")},
            "distributions": metadata,
        },
    )


def archive_suffix(tag: str) -> str:
    return ".tar.gz" if tag.startswith("linux-") else ".zip"


def make_archive(bundle: Path, archive: Path, tag: str) -> dict:
    refuse_existing(archive)
    if tag.startswith("windows-"):
        inventory(bundle, allow_symlinks=False)
        with zipfile.ZipFile(
            archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zipped:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    zipped.write(path, path.relative_to(bundle.parent).as_posix())
    elif tag.startswith("linux-"):
        with tarfile.open(archive, "x:gz", dereference=False) as tar:
            tar.add(bundle, arcname=bundle.name)
    elif tag.startswith("macos-"):
        # ditto retains executable permissions, resource forks and PyInstaller's framework symlinks.
        with tempfile.TemporaryDirectory(prefix="dsc-archive-", dir=archive.parent) as scratch:
            temporary = Path(scratch) / archive.name
            subprocess.run(
                [
                    "/usr/bin/ditto",
                    "-c",
                    "-k",
                    "--sequesterRsrc",
                    "--keepParent",
                    str(bundle),
                    str(temporary),
                ],
                check=True,
            )
            if temporary.stat().st_size >= MAX_ARCHIVE_BYTES:
                raise BuildError("archive is at least 2 GiB; refusing this release asset")
            copy_file(temporary, archive)
    else:
        raise BuildError(f"unknown archive platform: {tag}")
    if archive.stat().st_size >= MAX_ARCHIVE_BYTES:
        raise BuildError("archive is at least 2 GiB; refusing this release asset")
    return {"file": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive)}


def readme(model: str, executable: str, tag: str) -> str:
    text = f"""ChessDS {model} — standalone chess engine

Python is included. You do not need to install Python or download model files.

1. Extract the archive into a folder on your computer.
2. Keep the whole folder together, including _internal and models.
3. In your chess GUI, choose Add engine (or Manage engines), then select {executable}.
4. Start the engine. It predicts human moves; it does not search like Stockfish.

This build runs on {tag}. It uses your CPU and works without an internet connection.
The engine plays the most probable move by default. Ratings and other settings are
available in your GUI's engine options. The main models use the game's move history
when your GUI sends it; FEN models use a single position.

Code: GPL-3.0-or-later (LICENSE). Weights: CC BY 4.0 (LICENSE-weights),
attribution Albert W. Hamood / ChessDS. Editable source is in source/;
third-party notices and versions are in third_party_licenses/.
"""
    if tag.startswith("macos"):
        text += (
            "\nThis macOS build is ad-hoc signed and not notarized, so macOS blocks it until you\n"
            "clear the download quarantine for this folder. Open Terminal and run, with this\n"
            "folder's path:\n"
            "    xattr -dr com.apple.quarantine /path/to/this/folder\n"
            "Then add the engine in your GUI. The first analysis on a computer starts after a\n"
            "10-20 second check of the bundled libraries; later starts take under a second.\n"
            "Details: source/chessds-dsc/standalone/MACOS_FIRST_RUN.md\n"
        )
    elif tag.startswith("windows"):
        text += (
            "\nThis Windows build is unsigned. Windows may require your permission to open it.\n"
        )
    return text


def build(args: argparse.Namespace) -> Path:
    output = Path(args.output).expanduser().absolute()
    if output.resolve().is_relative_to(PACKAGE_ROOT.resolve()):
        raise BuildError("--output must be outside the package source tree")
    refuse_existing(output)
    weights = ensure_abs_file(args.weights, "--weights")
    weight_license = ensure_abs_file(args.weights_license, "--weights-license")
    python_license = (
        ensure_abs_file(args.python_license, "--python-license") if args.python_license else None
    )
    probe = probe_environment(args.model)
    system, tag = platform.system(), host_tag()
    validate_cpu(probe, system)
    config, data = validate_artifacts(weights, probe["entry"])
    identity = runtime_identity(probe, tag)
    runtime = (
        validate_runtime(Path(args.runtime_dir).expanduser().resolve(), identity)
        if args.runtime_dir
        else None
    )
    # Reserve the output atomically, only after validating every supplied artifact/runtime.
    output.mkdir(parents=True, exist_ok=False)
    if runtime is None:
        runtime = freeze_runtime(output / "runtime", identity)
    model, version = probe["entry"]["id"], probe["package_version"]
    bundle = output / f"{model}-{version}-{tag}"
    shutil.copytree(runtime, bundle, symlinks=True)
    old = bundle / ("DSC.exe" if system == "Windows" else "DSC")
    executable = model + (".exe" if system == "Windows" else "")
    refuse_existing(bundle / executable)
    old.rename(bundle / executable)
    copy_file(weights, bundle / "models" / weights.name)
    copy_file(config, bundle / "models" / config.name)
    contract = {
        "model_id": model,
        "artifact_id": data["model_id"],
        "mode": data["mode"],
        "weights": f"models/{weights.name}",
        "config": f"models/{config.name}",
        "weights_sha256": probe["entry"]["weights_sha256"],
        "config_sha256": probe["entry"]["config_sha256"],
    }
    write_json(bundle / "bundle.json", contract)
    copy_file(PACKAGE_ROOT / "LICENSE", bundle / "LICENSE")
    copy_file(weight_license, bundle / "LICENSE-weights")
    source_snapshot(bundle, probe)
    collect_licenses(bundle, probe, python_license)
    with (bundle / "README.txt").open("x", encoding="utf-8") as stream:
        stream.write(readme(model, executable, tag))
    files = inventory(bundle, allow_symlinks=system != "Windows")
    archive = output / (bundle.name + archive_suffix(tag))
    archived = make_archive(bundle, archive, tag)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - Python 3.10 support
        "status": "local release candidate; not published",
        "bundle": bundle.name,
        "contract": contract,
        "archive": archived,
        "files": files,
        "runtime_identity": identity,
        "runtime_reused_from": str(runtime.parent) if args.runtime_dir else None,
        "source_weights": str(weights),
        "source_config": str(config),
        "weights_license_sha256": sha256(weight_license),
        "signing": "ad-hoc; not notarized" if system == "Darwin" else "unsigned",
        "python": probe["python"],
        "dependencies": [
            {k: d[k] for k in ("name", "version", "source_urls")} for d in probe["dependencies"]
        ],
    }
    write_json(output / "build_receipt.json", receipt)
    print(f"Built local candidate: {archive}\nReceipt: {output / 'build_receipt.json'}", flush=True)
    return archive


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="dsc-150m")
    p.add_argument("--weights", required=True)
    p.add_argument("--weights-license", required=True)
    p.add_argument("--output", required=True, help="new output directory (must not exist)")
    p.add_argument(
        "--runtime-dir", help="existing runtime cache containing DSC/ and runtime_receipt.json"
    )
    p.add_argument(
        "--python-license", help="full path to Python's licence when the installation omits it"
    )
    return p


def main() -> None:
    try:
        build(parser().parse_args())
    except (BuildError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Build refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
