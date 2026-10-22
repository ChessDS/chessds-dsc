# Standalone UCI engine packaging

The player downloads an archive, extracts the whole folder, and selects the named executable
in their chess GUI. The archive contains Python, the inference libraries and one model. It
runs on the CPU and needs no Python installation, terminal commands, account or model download.
Keep the executable, `_internal` directory and `models` directory together.

These are maintainer build instructions. A successful build is a local candidate, not a public
release. Verify the extracted archive on its target system before offering it for download.

## Build on the target system

Use Python 3.11 in a clean virtual environment. On Windows and Linux, install a CPU-only PyTorch
wheel using the [official selector](https://pytorch.org/get-started/locally/), then install this
package and `standalone/requirements-build.txt`. On Apple silicon, use the normal macOS PyTorch
wheel; the launcher still selects CPU. Record the resolved runtime versions with the build.
Do not build from a training environment containing unrelated packages or GPU runtimes.

```sh
python -m pip install .
python -m pip install -r standalone/requirements-build.txt
python standalone/build.py --model dsc-150m --weights /path/to/dsc-150m-v1.safetensors --weights-license /path/to/LICENSE-weights --output /new/output/path
```

The output must be a new directory outside the package source tree.

The weights and same-stem JSON sidecar must be the unchanged files pinned in the package
registry. The builder refuses a mismatch. Repeat for another model with a new output path and
`--runtime-dir` pointing at the first output's runtime cache; its source, environment and file
hashes must still match. Each archive contains exactly one model.

The existing package presets remain unchanged: temperature 0, normal UCI options, and previous
positions enabled for main models. FEN models use no previous positions. A bundled launcher pins
the local model and CPU, resolves files relative to its executable rather than the GUI's working
directory, and fails clearly if the folder is incomplete or a configuration is corrupt.

## Verify the downloaded form

Extract the archive into a new directory with spaces in its name, preserving executable bits
and symlinks. On macOS use Archive Utility or `ditto -x -k`; on Linux use `tar`; on Windows use
Extract All. Run:

```sh
python standalone/verify.py --engine '/new path/dsc-150m-0.1.0-macos-arm64/dsc-150m' --receipt /new/verification.json
```

The verifier uses an unrelated working directory, fresh user/cache directories, and no inherited
Python paths. It checks UCI startup, readiness, the three output modes, legal moves, stop and
terminal-position behavior. Also test adding the extracted engine to a real chess GUI. For a
release, compare predictions against the Python package for the same positions and options.

PyInstaller builds are specific to the host OS and architecture. Validate Windows x64, Apple
silicon and Linux x64 builds separately. An Apple silicon build does not establish Intel Mac
support. Linux compatibility depends on the build system's glibc baseline.

## Distribution

The builder creates an archive, build receipt, source snapshot and third-party licence notices.
Preserve those with the exact archive checksum. Mac/Linux archives must retain symlinks; do not
repackage the runtime with a tool that replaces them with ordinary files. Build receipts describe
local candidates and must not be relabelled as release or platform validation evidence.

A macOS ad-hoc signature is sufficient for local testing, but public distribution needs Developer
ID signing and notarization to provide a smooth first launch. Windows signing and clean-machine
validation likewise belong to release preparation. Do not replace these checks with instructions
to disable OS security. This tooling performs no uploads, release creation or CI dispatch.

Implementation references: [PyInstaller bundles](https://pyinstaller.org/en/stable/operating-mode.html),
[runtime paths](https://pyinstaller.org/en/stable/runtime-information.html), and
[symlinks and standard streams](https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html).
