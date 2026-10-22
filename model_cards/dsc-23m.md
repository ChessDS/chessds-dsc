---
license: cc-by-4.0
library_name: chessds-dsc
tags: [chess, move-prediction, human-behavior, safetensors]
---

# DSC-23M

DSC-23M is a neural model that predicts human chess moves, with a focus on master and grandmaster play. It independently reimplements and extends Chessformer, the architecture used by Maia-3, with additional inputs for game context. Trained on high-rated games and fine-tuned on classical over-the-board play, it assigns a probability to every legal move from the position, ratings and game context.

This main model reads the current position and up to 7 previous positions. Pass a `chess.Board` with its move stack; a FEN alone supplies no previous positions. Use `dsc-23m-fen` for positions without a game record. DSC-23M also returns auxiliary win / draw / loss probabilities for the side to move.

| Identity | Value |
|---|---|
| Model | `dsc-23m`; artifact `dsc-23m-v1`; [ChessDS/DSC-23M](https://huggingface.co/ChessDS/DSC-23M), tag `v1` |
| Weights | `dsc-23m-v1.safetensors`, 90,172,120 bytes, fp32; sha256 `740d9393d041508b872643b0393569fdbb1662bb8a19e24159d980343d9f3189` |
| Config | `dsc-23m-v1.json`; sha256 `26d70ab23179cc3ec5f46d6511166015287161c7802d0ad686ab67678ac73328` |
| Architecture | 8 layers, width 512, 16 attention heads; 22,538,566 parameters |

FEN models are trained to predict from a single board position. Their name comes from FEN (Forsyth–Edwards Notation), the format used to describe that position. They use the position, side to move, castling rights, en passant square and halfmove clock, plus the players' ratings and game context you supply and, optionally, whether the position has occurred before, which a FEN does not record. They never use previous positions. Omitted ratings default to 2578 each; omitted game context defaults to classical and FIDE.

All variants share these defaults. Repetition flags come from a complete move stack or explicit `repetition=(seen_once_before, seen_twice_before)`. Without either, repetition is treated as absent. Ratings, time control and rating system can be supplied independently; venue is derived from the rating system.

## Training and evaluation

Pretraining: 1,284,001,792 sampled positions from Lichess games (October 2024–December 2025, both players 2200+, CC0) and The Week in Chess over-the-board games; moves made with under 30 seconds remaining were excluded as Lichess targets. Fine-tuning: 71,491,584 positions, two passes through 437,198 classical over-the-board games with both players 2400+. Evaluation games were excluded from training by fingerprint.

Every move is scored. eval91: 81,939 classical 2400+ games, 6,698,641 moves; two of 81,941 games omitted (one cannot be replayed, one starts from a set-up position). F0b: 1,260 classical 2400+ games, 110,215 moves, TWIC issues 1661–1665 (September 1–October 5, 2026), played after all training games. “Real” means each player's recorded rating.

| Set / ratings, history | Model or difference | top-1 | top-3 | log loss |
|---|---|---:|---:|---:|
| eval91 / real, history | DSC-23M | 60.55% | 88.14% | 1.1103 |
| eval91 / real, history | Maia-3 79M | 57.61% | 86.04% | 1.2210 |
| eval91 / real, history | Maia-3 23M | 56.39% | 85.06% | 1.2644 |
| eval91 / real, history | DSC − Maia-3 79M [95% CI] | +2.94 pp [+2.91, +2.97] | +2.10 pp [+2.08, +2.13] | -0.1107 [-0.1112, -0.1102] |
| F0b / real, history | DSC-23M | 59.55% | 87.44% | 1.1388 |
| F0b / real, history | Maia-3 79M | 56.75% | 85.34% | 1.2526 |
| F0b / real, history | Maia-3 23M | 55.50% | 84.29% | 1.2987 |

Maia-3 used the same positions, ratings and history; it has no time-control or rating-system input. FIDE ratings were passed without conversion. DSC was fine-tuned on classical over-the-board games; Maia-3 was trained on Lichess. Intervals are paired whole-game bootstrap 95% intervals (2,000 resamples, position-weighted).

The eval91 scores of DSC-23M and the Maia-3 eval91 references were computed in fp32 with TF32 off on the same GPU type (NVIDIA RTX PRO 4500). The DSC F0b scores were computed in fp32 on the training instances' NVIDIA A100 GPUs, and the Maia-3 F0b references on Apple silicon (MPS).

## Usage

Install: `pip install chessds-dsc`. Downloads are cached and sha256-verified.

```python
import chess
from chessds_dsc import DSC
model = DSC.load("dsc-23m")
board = chess.Board(); board.push_san("e4")
pred = model.predict(board, white_elo=2700, black_elo=2650)
```

UCI: `dsc-uci --model dsc-23m --use-uci-history`, or `dsc-23m` to play the most probable move. The selected model determines whether previous positions can be used. Main-model presets enable `--use-uci-history`, which feeds preceding positions from `position ... moves ...` to main-model weights. Without it, the same main weights receive no previous positions; they do not become the separately trained FEN model. FEN weights always ignore previous positions. A complete move list supplies repetition even without that flag.

`OutputMode=likelihood-wdl` (default) uses the same `score cp` / `wdl` line format as the Maia-3 engine, from this model's own WDL head after each candidate move, viewed from the mover's side, plus move probabilities. These are not engine evaluations. `likelihood` sends probabilities as `info string ... (P: ...)` lines; opt-in `probability-as-score` displays move probabilities in score fields. `MultiPV` truncates without rescaling; `Temperature` and `TopP` affect the move played, not the reported probabilities.

- Evaluated on classical games with both players 2400+; performance below that level and on rapid, blitz or bullet is unmeasured.
- The raw WDL head is an auxiliary training output. Do not rely on it as a standalone game-result forecast; in our 23M tests it was less accurate than dedicated result models.
- No search: depth 1; clock and movetime settings do not change the prediction.

ChessDS's full result and move-prediction models combine neural outputs with Stockfish evaluations and other features. The full move predictor achieves higher accuracy than standalone neural move prediction. These full models are separate from this package.

Weights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), attribution Albert W. Hamood / ChessDS. [Inference code](https://github.com/ChessDS/chessds-dsc): GPL-3.0-or-later, including its python-chess dependency.

Cite Monroe, Eilender, Chalmers, Tang and Anderson, *Chessformer: A Unified Architecture for Chess Modeling*, ICLR 2026, arXiv:2605.19091; and Albert W. Hamood, *DSC: Human Chess Move Prediction Models*, ChessDS, 2026, [chessds.com](https://chessds.com/).

No Maia-3 code, weights or outputs were used in developing or training this model. Maia-3 was run only for the comparison rows.
