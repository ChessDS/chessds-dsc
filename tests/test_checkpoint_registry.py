from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path

import pytest
import torch
from chessds_dsc import registry
from chessds_dsc.checkpoint import FORMAT, load_model, read_config, save_model, sha256_file
from conftest import tiny_model

FULL = {"elo": 2578, "time_control": "classical", "rating_system": "fide"}


def test_save_load_round_trip(tmp_path):
    model = tiny_model(3)
    stem = tmp_path / "tiny"
    config = save_model(
        model, stem, model_id="tiny", mode="history", defaults=FULL, extra={"size": "t"}
    )
    weights = stem.with_suffix(".safetensors")
    assert config["format"] == FORMAT and config["parameters"] == sum(
        p.numel() for p in model.parameters()
    )
    assert config["weights_sha256"] == sha256_file(weights)
    back, cfg = load_model(weights)
    for k, v in model.state_dict().items():
        assert torch.equal(back.state_dict()[k], v)
    assert cfg["size"] == "t" and not back.training

    # Without the sidecar the embedded metadata still carries the architecture.
    stem.with_suffix(".json").unlink()
    cfg2 = read_config(weights)
    assert cfg2["architecture"] == config["architecture"] and cfg2["mode"] == "history"
    back2, _ = load_model(weights)
    assert torch.equal(back2.policy.q.weight, model.policy.q.weight)


def test_tampered_weights_are_rejected(tmp_path):
    stem = tmp_path / "tiny"
    save_model(tiny_model(), stem, model_id="tiny", mode="strict", defaults=FULL)
    weights = stem.with_suffix(".safetensors")
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        load_model(weights)
    load_model(weights, verify=False)  # the bytes still parse


def test_bad_format_or_mode_rejected(tmp_path):
    stem = tmp_path / "tiny"
    save_model(tiny_model(), stem, model_id="tiny", mode="strict", defaults=FULL)
    cfg_path = stem.with_suffix(".json")
    cfg = json.loads(cfg_path.read_text())
    cfg_path.write_text(json.dumps({**cfg, "format": "other"}))
    with pytest.raises(ValueError, match="format"):
        read_config(stem.with_suffix(".safetensors"))
    cfg_path.write_text(json.dumps({**cfg, "mode": "mystery"}))
    with pytest.raises(ValueError, match="mode"):
        read_config(stem.with_suffix(".safetensors"))


@pytest.fixture
def served_registry(tmp_path, monkeypatch):
    """A tiny model served over HTTP in the default layout, <base>/<repo>/resolve/<revision>/<file>,
    registered under 'tiny-served'."""
    root = tmp_path / "www"
    stem = root / "tiny-served" / "resolve" / "v1" / "tiny-served-rc1"
    save_model(tiny_model(5), stem, model_id="tiny-served-rc1", mode="history", defaults=FULL)
    weights, config = stem.with_suffix(".safetensors"), stem.with_suffix(".json")
    entry = registry.ModelEntry(
        id="tiny-served",
        aliases=("tiny",),
        size="t",
        mode="history",
        parameters=1,
        weights=weights.name,
        config=config.name,
        weights_sha256=sha256_file(weights),
        config_sha256=sha256_file(config),
        description="test model",
    )
    monkeypatch.setattr(registry, "MODELS", (entry,))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    cache = tmp_path / "cache"
    yield entry, base, cache, root
    server.shutdown()


def test_download_verify_cache_and_offline(served_registry, monkeypatch):
    entry, base, cache, root = served_registry
    assert registry.resolve("TINY") is entry and registry.resolve("tiny-served") is entry
    with pytest.raises(KeyError):
        registry.resolve("dsc-500m")
    with pytest.raises(FileNotFoundError):
        registry.fetch(entry, cache_dir_=cache, base_url_=base, local_files_only=True)
    with pytest.raises(RuntimeError, match="download location"):
        monkeypatch.delenv(registry.ENV_BASE_URL, raising=False)
        monkeypatch.setattr(registry, "DEFAULT_BASE_URL", None)  # no built-in location
        registry.fetch(entry, cache_dir_=cache)
    w, c = registry.fetch(entry, cache_dir_=cache, base_url_=base, log=lambda m: None)
    assert w == cache / "tiny-served" / entry.weights and c.exists()
    assert sha256_file(w) == entry.weights_sha256
    model, cfg = load_model(w, c)
    assert cfg["model_id"] == "tiny-served-rc1"
    # Second fetch is served from the cache even with no network location.
    w2, _ = registry.fetch(entry, cache_dir_=cache, local_files_only=True)
    assert w2 == w
    # A corrupt cached file is replaced.
    w.write_bytes(b"junk")
    w3, _ = registry.fetch(entry, cache_dir_=cache, base_url_=base, log=lambda m: None)
    assert sha256_file(w3) == entry.weights_sha256
    # The environment variable configures the location too.
    monkeypatch.setenv(registry.ENV_BASE_URL, base)
    registry.fetch(entry, cache_dir_=cache, force_download=True, log=lambda m: None)


def test_corrupt_download_is_not_installed(served_registry):
    entry, base, cache, root = served_registry
    (root / "tiny-served" / "resolve" / "v1" / entry.weights).write_bytes(b"not the weights")
    with pytest.raises(ValueError, match="sha256"):
        registry.fetch(entry, cache_dir_=cache, base_url_=base, log=lambda m: None)
    assert not (cache / "tiny-served" / entry.weights).exists()
    assert (
        not list(Path(cache / "tiny-served").glob("*.part"))
        if (cache / "tiny-served").exists()
        else True
    )


def test_registry_has_no_private_models():
    for entry in registry.MODELS:
        assert "500" not in entry.id and entry.size in ("23m", "79m", "150m")
        assert len(entry.weights_sha256) == 64 and len(entry.config_sha256) == 64
    names = [f for e in registry.MODELS for f, _ in e.files]
    assert len(names) == len(set(names)), "artifact file names must be unique across models"


def test_default_layout_uses_repo_and_revision(monkeypatch):
    monkeypatch.delenv(registry.ENV_URL_TEMPLATE, raising=False)
    entry = registry.ModelEntry(
        id="m-h",
        aliases=(),
        size="23m",
        mode="history",
        parameters=1,
        weights="m-h-rc1.safetensors",
        config="m-h-rc1.json",
        weights_sha256="0" * 64,
        config_sha256="0" * 64,
        description="",
        repo="M-Repo",
        revision="v7",
    )
    hf = "https://huggingface.co/Org"
    assert (
        registry.file_url(entry, entry.weights, hf) == f"{hf}/M-Repo/resolve/v7/m-h-rc1.safetensors"
    )
    mirror = "https://mirror.example/Org"  # a mirror copies the same paths; only the base differs
    assert (
        registry.file_url(entry, entry.config, mirror) == f"{mirror}/M-Repo/resolve/v7/m-h-rc1.json"
    )
    no_repo = registry.ModelEntry(**{**entry.__dict__, "repo": ""})
    assert registry.file_url(no_repo, "f", hf) == f"{hf}/m-h/resolve/v7/f"


def test_url_template_override(served_registry, monkeypatch):
    entry, base, cache, root = served_registry
    monkeypatch.delenv(registry.ENV_BASE_URL, raising=False)
    monkeypatch.setattr(registry, "DEFAULT_BASE_URL", None)  # no built-in location
    # A template without {base} needs no base URL.
    monkeypatch.setenv(registry.ENV_URL_TEMPLATE, base + "/{repo}/resolve/{revision}/{file}")
    assert (
        registry.file_url(entry, entry.weights) == f"{base}/{entry.id}/resolve/v1/{entry.weights}"
    )
    w, _ = registry.fetch(entry, cache_dir_=cache, log=lambda m: None)
    assert sha256_file(w) == entry.weights_sha256
    monkeypatch.setenv(registry.ENV_URL_TEMPLATE, "{base}/x/{id}/{file}")
    with pytest.raises(RuntimeError, match="download location"):
        registry.file_url(entry, entry.weights)
    assert (
        registry.file_url(entry, entry.weights, "http://h")
        == f"http://h/x/{entry.id}/{entry.weights}"
    )


def test_metadata_only_load_keeps_defaults_and_rejects_other_contracts(tmp_path):
    import chess
    from chessds_dsc import DSC
    from chessds_dsc.encoding import ENCODING_VERSION

    stem = tmp_path / "tiny"
    custom = {"elo": 2000, "time_control": "blitz", "rating_system": "lichess"}
    save_model(tiny_model(), stem, model_id="tiny", mode="strict", defaults=custom)
    weights = stem.with_suffix(".safetensors")
    stem.with_suffix(".json").unlink()
    cfg = read_config(weights)  # the weights file alone carries the whole input contract
    assert cfg["defaults"] == custom and cfg["encoding_version"] == ENCODING_VERSION
    assert cfg["parameters"] > 0 and cfg["mode"] == "strict"
    dsc = DSC.load(checkpoint_path=weights, device="cpu")
    pred = dsc.predict(chess.Board())
    assert dsc.default_elo == 2000 and pred.inputs.elo_mover == 2000
    assert (pred.inputs.time_control, pred.inputs.rating_system) == ("blitz", "lichess")

    std = {"elo": 2578, "time_control": "classical", "rating_system": "fide"}
    save_model(tiny_model(), stem, model_id="tiny", mode="strict", defaults=std)
    cfg_path = stem.with_suffix(".json")
    cfg = json.loads(cfg_path.read_text())
    cfg_path.write_text(json.dumps({**cfg, "encoding_version": "dsc-encoding-v0"}))
    with pytest.raises(ValueError, match="encoding version"):
        read_config(weights)
    cfg_path.write_text(json.dumps({k: v for k, v in cfg.items() if k != "defaults"}))
    with pytest.raises(ValueError, match="defaults"):
        read_config(weights)
    with pytest.raises(ValueError, match="defaults"):
        save_model(tiny_model(), stem, model_id="tiny", mode="strict", defaults={"elo": 1})
