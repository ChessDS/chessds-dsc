---
license: cc-by-4.0
library_name: chessds-dsc
tags: [chess, move-prediction, human-behavior, safetensors]
---

# DSC-150M-FEN

DSC-150M-FEN is a neural model that predicts human chess moves, with a focus on master and grandmaster play. It independently reimplements and extends Chessformer, the architecture introduced by the Maia-3 team of Monroe, Eilender, Chalmers, Tang and Anderson ([paper](https://arxiv.org/abs/2605.19091), [code](https://github.com/CSSLab/maia3)), with additional inputs for game context. Trained on high-rated games and fine-tuned on classical over-the-board play, it assigns a probability to every legal move from the position, ratings and game context.

This FEN model is trained for single-position input and never uses previous positions. The main model `dsc-150m` accepts up to 7 previous positions. DSC-150M-FEN also returns auxiliary win / draw / loss probabilities for the side to move.

| Identity | Value |
|---|---|
| Model | `dsc-150m-fen`; artifact `dsc-150m-fen-v1`; [ChessDS/DSC-150M-FEN](https://huggingface.co/ChessDS/DSC-150M-FEN), tag `v1` |
| Weights | `dsc-150m-fen-v1.safetensors`, 603,663,000 bytes, fp32; sha256 `6392b549d635b85b2a7615f5fffeadf29aebc67f6da869352b4411ff558c19d3` |
| Config | `dsc-150m-fen-v1.json`; sha256 `c0d3462dbd6ade150ed81c9556329ce8218050eb2fe06c34b4a9062b8d3f1944` |
| Architecture | 16 layers, width 1024, 32 attention heads; 150,907,206 parameters |

FEN models are trained to predict from a single board position. Their name comes from FEN (Forsyth–Edwards Notation), the format used to describe that position. They use the position, side to move, castling rights, en passant square and halfmove clock, plus the players' ratings and game context you supply and, optionally, whether the position has occurred before, which a FEN does not record. They never use previous positions. Omitted ratings default to 2578 for each player, the average rating in the GM-level classical fine-tuning set. Omitted game context defaults to classical and FIDE.

All variants share these defaults. Repetition flags come from a complete move stack or explicit `repetition=(seen_once_before, seen_twice_before)`. Without either, repetition is treated as absent. Ratings, time control and rating system can be supplied independently; venue is derived from the rating system.

## Training and evaluation

Pretraining: 1,620,000,768 sampled positions from Lichess games (October 2024–December 2025, both players 2200+, CC0) and The Week in Chess over-the-board games; moves made with under 30 seconds remaining were excluded as Lichess targets. Fine-tuning: 71,491,584 positions, two passes through 437,051 classical over-the-board games with both players 2400+. Evaluation games were excluded from training by fingerprint.

Every move is scored. eval91: 81,939 classical 2400+ games, 6,698,641 moves; two of 81,941 games omitted (one cannot be replayed, one starts from a set-up position). F0b: 1,260 classical 2400+ games, 110,215 moves, TWIC issues 1661–1665 (September 1–October 5, 2026), played after all training games. “Real” means each player's recorded rating.

| Set / ratings, history | Model or difference | top-1 | top-3 | log loss |
|---|---|---:|---:|---:|
| eval91 / real, no history | DSC-150M-FEN | 61.84% | 89.19% | 1.0633 |
| eval91 / real, no history | Maia-3 79M | 55.73% | 84.95% | 1.2798 |
| eval91 / real, no history | Maia-3 23M | 54.11% | 83.78% | 1.3319 |
| eval91 / real, no history | DSC − Maia-3 79M [95% CI] | +6.11 pp [+6.07, +6.14] | +4.24 pp [+4.21, +4.26] | -0.2165 [-0.2172, -0.2158] |
| eval91 / 2578, no history | DSC-150M-FEN | 61.76% | 89.14% | 1.0663 |
| eval91 / 2578, no history | Maia-3 79M | 55.91% | 85.09% | 1.2743 |
| eval91 / 2578, no history | Maia-3 23M | 54.15% | 83.86% | 1.3282 |
| eval91 / 2578, no history | DSC − Maia-3 79M [95% CI] | +5.85 pp [+5.82, +5.89] | +4.05 pp [+4.03, +4.08] | -0.2080 [-0.2088, -0.2073] |
| F0b / 2578, no history | DSC-150M-FEN | 60.85% | 88.61% | 1.0899 |
| F0b / 2578, no history | Maia-3 79M | 55.14% | 84.27% | 1.3065 |
| F0b / 2578, no history | Maia-3 23M | 53.26% | 83.17% | 1.3619 |

Maia-3 used the same positions, ratings and history; it has no time-control or rating-system input. FIDE ratings were passed without conversion. DSC was fine-tuned on classical over-the-board games; Maia-3 was trained on Lichess. Intervals are paired whole-game bootstrap 95% intervals (2,000 resamples, position-weighted). Fixed-2578 rows also use classical / FIDE / over the board for DSC. F0b has no recorded-rating evaluation for this model.

The eval91 scores of DSC-150M-FEN and the Maia-3 eval91 references were computed in fp32 with TF32 off on the same GPU type (NVIDIA RTX PRO 4500). The DSC F0b scores were computed in fp32 on the training instances' NVIDIA A100 GPUs, and the Maia-3 F0b references on Apple silicon (MPS).

## Usage

Install from GitHub while PyPI publication is pending: `pip install "git+https://github.com/ChessDS/chessds-dsc.git@v0.1.0-rc1"`. Model downloads are cached and sha256-verified. [Standalone UCI downloads](https://github.com/ChessDS/chessds-dsc/releases/tag/v0.1.0-rc1) include the runtime and model.

```python
import chess
from chessds_dsc import DSC
model = DSC.load("dsc-150m-fen")
board = chess.Board(); board.push_san("e4")
pred = model.predict(board, white_elo=2700, black_elo=2650)
```

UCI: `dsc-uci --model dsc-150m-fen`, or `dsc-150m-fen` to play the most probable move. The selected model determines whether previous positions can be used. Main-model presets enable `--use-uci-history`, which feeds preceding positions from `position ... moves ...` to main-model weights. Without it, the same main weights receive no previous positions; they do not become the separately trained FEN model. FEN weights always ignore previous positions. A complete move list supplies repetition even without that flag.

`OutputMode=likelihood-wdl` (default) uses the same `score cp` / `wdl` line format as the Maia-3 engine, from this model's own WDL head after each candidate move, viewed from the mover's side, plus move probabilities. These are not engine evaluations. `likelihood` sends probabilities as `info string ... (P: ...)` lines; opt-in `probability-as-score` displays move probabilities in score fields. `MultiPV` truncates without rescaling; `Temperature` and `TopP` affect the move played, not the reported probabilities.

- Evaluated on classical games with both players 2400+; performance below that level and on rapid, blitz or bullet is unmeasured.
- The raw WDL head is an auxiliary training output. Do not rely on it as a standalone game-result forecast; in our 23M tests it was less accurate than dedicated result models.
- No search: depth 1; clock and movetime settings do not change the prediction.

ChessDS's full result and move-prediction models combine neural outputs with Stockfish evaluations and other features. The full move predictor achieves higher accuracy than standalone neural move prediction. These full models are separate from this package.

Weights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), attribution Albert W. Hamood / ChessDS. [Inference code](https://github.com/ChessDS/chessds-dsc): GPL-3.0-or-later, including its python-chess dependency.

No Maia-3 code, weights or outputs were used in developing or training this model. The released Maia-3 weights ([UofTCSSLab/Maia3-79M](https://huggingface.co/UofTCSSLab/Maia3-79M), [Maia3-23M](https://huggingface.co/UofTCSSLab/Maia3-23M)) were run only for the comparison rows; the Maia-3 code (AGPL-3.0) is neither included in nor required by the inference package.

## Citation

DSC builds on the Chessformer architecture introduced by Maia-3 ([paper](https://arxiv.org/abs/2605.19091), [code](https://github.com/CSSLab/maia3), [models](https://huggingface.co/collections/MaiaChess/maia3), [maiachess.com](https://maiachess.com)). If you use this model, please cite the Chessformer paper, as its authors ask, and DSC:

```bibtex
@inproceedings{monroe2026chessformer,
title={Chessformer: A Unified Architecture for Chess Modeling},
author={Daniel Monroe and George Eilender and Philip Chalmers and Zhenwei Tang and Ashton Anderson},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=2ltBRzEHyd}
}
```

```bibtex
@misc{hamood2026dsc,
title={DSC: Human Chess Move Prediction Models},
author={Albert W. Hamood},
year={2026},
publisher={ChessDS},
howpublished={\url{https://chessds.com/}},
note={Weights: \url{https://huggingface.co/ChessDS}; code: \url{https://github.com/ChessDS/chessds-dsc}}
}
```
