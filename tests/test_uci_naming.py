"""Public UCI names stay canonical while legacy aliases remain accepted.

These checks never instantiate a model or read a checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from chessds_dsc import uci
from chessds_dsc.registry import MODELS, format_model_list, resolve

ALIASES = [(entry.id, alias) for entry in MODELS for alias in entry.aliases]


@pytest.mark.parametrize("model_id,alias", ALIASES, ids=[alias for _, alias in ALIASES])
def test_alias_uci_identity_is_canonical_before_loading(model_id, alias, monkeypatch, capsys):
    def unexpected_load(*args, **kwargs):
        pytest.fail("identifying an engine must not load its weights")

    monkeypatch.setattr(uci.DSC, "load", staticmethod(unexpected_load))
    args = uci.parse_args(["--model", alias])
    assert args.model == model_id
    engine = uci.Engine(args)
    engine.cmd_uci()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"id name ChessDS DSC {model_id}"
    assert lines[-1] == "uciok"
    assert engine.dsc is None


def test_alias_resolution_remains_available_case_insensitively():
    for entry in MODELS:
        for alias in entry.aliases:
            assert resolve(alias) is entry
            assert resolve("  " + alias.upper() + "  ") is entry


def test_model_listing_uses_only_canonical_names_and_main_fen_labels():
    listing = format_model_list()
    assert "aliases:" not in listing
    assert "strict" not in listing.lower()
    rows = [line.split() for line in listing.splitlines() if line.startswith("dsc-")]
    assert [row[0] for row in rows] == [entry.id for entry in MODELS]
    for entry, row in zip(MODELS, rows, strict=True):
        assert row[2] == ("FEN" if entry.mode == "strict" else "main")
    # The short forms such as '23m' remain useful size labels; retired engine
    # suffixes must not appear as names anywhere in the listing.
    for _, alias in ALIASES:
        if alias.startswith("dsc-"):
            assert alias not in listing.split()


@pytest.mark.parametrize("entry", MODELS, ids=[entry.id for entry in MODELS])
def test_ready_log_uses_public_variant_and_loaded_id_keeps_artifact_version(
    entry, monkeypatch, capsys
):
    artifact_id = entry.weights.removesuffix(".safetensors")
    fake_model = SimpleNamespace(model_id=artifact_id, mode=entry.mode, device="cpu")
    calls = []

    def fake_load(name, **kwargs):
        calls.append((name, kwargs))
        return fake_model

    monkeypatch.setattr(uci.DSC, "load", staticmethod(fake_load))
    engine = uci.Engine(uci.parse_args(["--model", entry.id, "--device", "cpu"]))
    assert engine.ensure_loaded() is fake_model
    assert engine.ensure_loaded() is fake_model
    assert len(calls) == 1 and calls[0][0] == entry.id
    engine.cmd_uci()
    captured = capsys.readouterr()
    label = "FEN" if entry.mode == "strict" else "main"
    assert f"{artifact_id} ready ({label}, cpu)" in captured.err.splitlines()
    assert "strict" not in captured.err
    assert captured.out.splitlines()[0] == f"id name ChessDS DSC {artifact_id}"


@pytest.mark.parametrize("checkpoint", [None, "local-experiment.safetensors"])
def test_unknown_custom_model_name_is_preserved(checkpoint):
    args = ["--model", "local-experiment"]
    if checkpoint is not None:
        args += ["--checkpoint-path", checkpoint]
    parsed = uci.parse_args(args)
    assert parsed.model == "local-experiment"
    assert parsed.checkpoint_path == checkpoint
