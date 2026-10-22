from __future__ import annotations

import chess
import pytest
import torch
from chessds_dsc.checkpoint import save_model
from chessds_dsc.model import DSCConfig, DSCModel

TINY = DSCConfig(
    width=64,
    n_layers=2,
    elo_dim=16,
    cond_dim=8,
    gab_compress=4,
    gab_hidden=16,
    gab_head_dim=16,
    wdl_hidden=16,
)


def boards_after(line: str) -> list[chess.Board]:
    """Boards along a SAN line, current first then most recent previous (encoder convention)."""
    board = chess.Board()
    boards = [board.copy()]
    for san in line.split():
        board.push_san(san)
        boards.append(board.copy())
    return boards[::-1]


def tiny_model(seed: int = 0) -> DSCModel:
    torch.manual_seed(seed)
    model = DSCModel(TINY)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.2)
    return model.eval()


@pytest.fixture(scope="session")
def tiny_history_model_path(tmp_path_factory) -> str:
    stem = tmp_path_factory.mktemp("models") / "tiny-h"
    save_model(
        tiny_model(0),
        stem,
        model_id="tiny-h",
        mode="history",
        defaults={"elo": 2578, "time_control": "classical", "rating_system": "fide"},
    )
    return str(stem.with_suffix(".safetensors"))


@pytest.fixture(scope="session")
def tiny_strict_model_path(tmp_path_factory) -> str:
    stem = tmp_path_factory.mktemp("models") / "tiny-s"
    save_model(
        tiny_model(1),
        stem,
        model_id="tiny-s",
        mode="strict",
        defaults={"elo": 2578, "time_control": "classical", "rating_system": "fide"},
    )
    return str(stem.with_suffix(".safetensors"))
