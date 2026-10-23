# ChessDS DSC

DSC is a family of neural networks that predict human chess moves, with a focus on master and grandmaster play. It independently reimplements and extends Chessformer, the architecture used by Maia-3, adding inputs for game context and available history. The models are trained on high-rated games and fine-tuned on classical over-the-board play, assigning a probability to every legal move from the position, ratings and game context.

On 81,939 held-out classical games between players rated 2400+, DSC-150M predicts the move played 62.74% of the time, with the played move among its top three predictions 89.63% of the time, using recorded ratings and game history.

The three model sizes—23M, 79M and 150M—are named for their approximate parameter counts in millions. Each size has a main version that can read up to seven previous positions and a FEN version trained to work from the information in a single board position. Both use the ratings and game context you supply.

Version 0.1.0 provides a Python API and a UCI engine.
Code: [ChessDS/chessds-dsc](https://github.com/ChessDS/chessds-dsc). Weights: [Hugging Face](https://huggingface.co/ChessDS), downloaded from tag `v1`, verified by SHA-256 and cached.

## Accuracy

Every move of eval91: 81,939 held-out classical over-the-board games between players rated 2400+, with recorded ratings. Main models receive history; FEN models do not. Maia-3 79M uses its released weights on the same moves and inputs.

| Model | History | Top-1 | Top-3 | Log loss |
|---|---|---:|---:|---:|
| DSC-150M | yes | 62.74% | 89.63% | 1.038 |
| DSC-79M | yes | 61.81% | 89.01% | 1.069 |
| DSC-23M | yes | 60.55% | 88.14% | 1.110 |
| DSC-150M-FEN | no | 61.84% | 89.19% | 1.063 |
| DSC-79M-FEN | no | 60.97% | 88.60% | 1.091 |
| DSC-23M-FEN | no | 59.67% | 87.69% | 1.134 |
| Maia-3 79M | yes | 57.61% | 86.04% | 1.221 |
| Maia-3 79M | no | 55.73% | 84.95% | 1.280 |

DSC was fine-tuned on classical over-the-board games; Maia-3 was trained on Lichess games and received FIDE ratings unconverted. The model cards give intervals, fixed-rating results and F0b, a held-out set played after all training data.

ChessDS's full result and move-prediction models combine neural outputs with Stockfish evaluations and other features. The full move predictor achieves higher accuracy than standalone neural move prediction. These full models are separate from this package.

## Install and quickstart

[Standalone UCI downloads](https://github.com/ChessDS/chessds-dsc/releases/tag/v0.1.0-rc1) are available for Windows x64, Linux x64, Apple silicon Mac and Intel Mac. Each includes one model and its runtime; extract the whole folder and select its named executable in your chess GUI. No separate Python installation is required.

For the Python API, install from GitHub while PyPI publication is pending:

```bash
pip install "git+https://github.com/ChessDS/chessds-dsc.git@v0.1.0-rc1"
```

Requires Python 3.10+, PyTorch 2.1+, python-chess, numpy and safetensors. CPU, CUDA and Apple MPS are supported; pass `device="cpu"` to `DSC.load` to select CPU.

```python
from chessds_dsc import DSC
model = DSC.load("dsc-79m-fen")
fen = "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"
pred = model.predict(fen, white_elo=2700, black_elo=2650)
print(pred.moves)
```

`predict` accepts a FEN or `chess.Board`. `pred.best` is the most probable move; `pred.wdl` is the auxiliary head's WDL for the side to move; `pred.inputs` records the resolved inputs. `predict_batch([PositionInput(...), ...])` scores a batch. See [examples](https://github.com/ChessDS/chessds-dsc/tree/main/examples).

To sample a move in Python, apply the optional controls after prediction:
```python
from chessds_dsc.sampling import select_move
move = select_move(pred.moves, temperature=0.8, top_p=0.95)
```

## Models

| Model | id / UCI preset | Parameters | Previous positions | Weights |
|---|---|---:|---|---:|
| [DSC-23M](https://github.com/ChessDS/chessds-dsc/blob/main/model_cards/dsc-23m.md) | `dsc-23m` | 22.5M | up to 7 | 90 MB |
| [DSC-23M-FEN](https://github.com/ChessDS/chessds-dsc/blob/main/model_cards/dsc-23m-fen.md) | `dsc-23m-fen` | 22.5M | none | 90 MB |
| [DSC-79M](https://github.com/ChessDS/chessds-dsc/blob/main/model_cards/dsc-79m.md) | `dsc-79m` | 77.1M | up to 7 | 308 MB |
| [DSC-79M-FEN](https://github.com/ChessDS/chessds-dsc/blob/main/model_cards/dsc-79m-fen.md) | `dsc-79m-fen` | 77.1M | none | 308 MB |
| [DSC-150M](https://github.com/ChessDS/chessds-dsc/blob/main/model_cards/dsc-150m.md) | `dsc-150m` | 150.9M | up to 7 | 604 MB |
| [DSC-150M-FEN](https://github.com/ChessDS/chessds-dsc/blob/main/model_cards/dsc-150m-fen.md) | `dsc-150m-fen` | 150.9M | none | 604 MB |

FEN models are trained to predict from a single board position. Their name comes from FEN (Forsyth–Edwards Notation), the format used to describe that position. They use the position, side to move, castling rights, en passant square and halfmove clock, plus the players' ratings and game context you supply and, optionally, whether the position has occurred before, which a FEN does not record. They never use previous positions. Omitted ratings default to 2578 for each player, the average rating in the GM-level classical fine-tuning set. Omitted game context defaults to classical and FIDE.
The main and FEN variants have separately trained weights. Model-id aliases (`-h`, `-history`, `-s`, `-strict`, and bare sizes such as `23m`) remain; the FEN variant's API `mode` is `strict`.

## Inputs and history

| Input | Default | Accepted values / source |
|---|---|---|
| `white_elo`, `black_elo` | 2578 each | Players' ratings; omitting them predicts for two strong grandmasters. |
| `rating_system` | `"fide"` | `fide`, `lichess`, `other`; use `lichess` with Lichess ratings. |
| `time_control` | `"classical"` | `classical`, `rapid`, `blitz`, `bullet`. |
| `venue` | derived | `otb` for FIDE ratings, `online` otherwise; may be supplied. |
| Previous positions | board move stack | Main models use up to 7; FEN models never use them. |
| `repetition` | complete move stack | `(seen_once_before, seen_twice_before)`; assumed absent when unknown. |

In Python, main models use available history (`use_history=None`); `False` ignores it and `True` requires it. A FEN has no history. Repetition is inferred only from a move stack covering the whole game from the standard start; pass `repetition=(True, False)` when known separately.
The selected model determines whether it can use previous positions. Main UCI presets enable history automatically; with generic `dsc-uci`, `--use-uci-history` supplies previous positions from the GUI's move list to a main model. Without the flag, a main model receives no previous positions. The flag never switches weights, and FEN models ignore previous positions even when it is set. Choose a FEN model for predictions without history. UCI repetition comes from `position startpos moves ...`, even without that flag; `position fen ...` assumes no repetition.

## UCI output

`dsc-uci --model dsc-150m --use-uci-history` starts the engine. `OutputMode` / `--output-mode` selects:

- `likelihood-wdl` (default): move probabilities plus the same `info ... multipv N ... score cp ... wdl ...` line format the Maia-3 engine sends. The values come from DSC's own auxiliary WDL head after each candidate move, viewed from the mover's side. `score cp` is win minus loss in permille, not an engine evaluation or the move's probability.
- `likelihood`: move probabilities as `info string <move> (P: ...%)`, context and summary lines, and score-free `multipv` lines; one network evaluation.
- `probability-as-score` (opt-in): also writes probability × 10000 as `score cp`, negative for Black to move, so GUIs displaying White's perspective show +41.73 for 41.73%. For viewing: evaluation graphs and adjudication would treat it as an evaluation.

| Optional control (GUI / command line) | Default | Effect |
|---|---|---|
| `Temperature` / `--temperature` | 0 in named presets; 1 in `dsc-uci` | 0 picks the most probable move; 1 samples from the predicted probabilities. Between 0 and 1 favors likely moves more strongly; above 1 spreads the choices. |
| `TopP` / `--top-p` | 1 | 1 keeps all moves; lower values restrict sampling to the most likely moves, after temperature scaling. Ignored at temperature 0. |
| `MultiPV` / `--multipv` | 5 | Displays 1–20 candidates; does not restrict the moves that can be sampled. |
| `--seed` (command line only) | 42 | Repeats the sampling stream for the same inputs and call sequence. |

For example, `dsc-23m --temperature 0.8 --top-p 0.95 --seed 42` enables sampling in a preset. These controls leave the reported move probabilities unchanged. Python's `select_move` defaults to `temperature=1`, `top_p=1`; pass `rng=numpy.random.default_rng(42)` for a seeded sampler.
`go searchmoves` restricts candidates and sampling. There is no search; `go infinite` and `go ponder` hold the computed answer until `stop` or `ponderhit`.

## In a chess GUI

Set the engine command to an installed preset such as `dsc-150m` or `dsc-79m-fen`, or to `dsc-uci` with `--model dsc-150m --use-uci-history`. On macOS/Linux, `which dsc-150m` gives the path; on Windows, look in the Python environment's `Scripts` folder.
Run `dsc-cache --model dsc-150m` before the first game; otherwise the GUI waits for the initial download. Set `Elo`, `SelfElo`, `OppoElo`, `TimeControl`, `RatingSystem`, `OutputMode`, `Temperature`, `TopP` and `MultiPV` in the engine options. These commands are for the Python package. Standalone downloads include their runtime and model. All four platform packages passed extracted-engine UCI checks; consumer-machine GUI checks remain. Mac packages are ad-hoc signed and unnotarized, and Windows packages are unsigned.

## Coming from Maia-3

For a history-free Maia-3 setup, use a FEN model. For history, use a main preset or a main model with `--use-uci-history`; this flag supplies the history to the selected model.

| Maia-3 | DSC |
|---|---|
| `maia3-uci --model maia3-23m` (no history) | `dsc-uci --model dsc-23m-fen` |
| `maia3-uci --model maia3-23m --use-uci-history` | `dsc-uci --model dsc-23m --use-uci-history` |
| `maia3-23m` preset | `dsc-23m-fen` without history; `dsc-23m` with history. Both play the most probable move. |
| `maia3-uci --checkpoint-path FILE` | `dsc-uci --checkpoint-path FILE.safetensors`; DSC weights required |
| `maia3-uci --device cpu` | `dsc-uci --device cpu` |
| `maia3-cache --model maia3-23m` | `dsc-cache --model dsc-23m-fen` (or `dsc-23m` for history) |

## Licences and citation

Code: [GPL-3.0-or-later](https://github.com/ChessDS/chessds-dsc/blob/main/LICENSE), including when distributed in combination with python-chess, which uses the same licence. Redistribution must meet the GPL's terms, including its corresponding-source requirements; installing a dependency with pip does not remove those obligations.
Weights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with attribution to Albert W. Hamood / ChessDS, a licence link and an indication of changes. The weights can be used independently of this package.
DSC independently reimplements and extends the Chessformer architecture with inputs for time control, rating system, venue and history availability. It was trained on Lichess games (2200+, CC0) and The Week in Chess, then fine-tuned on classical over-the-board games. No Maia-3 code, weights or outputs were used to build or train DSC; Maia-3 was run only for the comparison results.
Cite Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang and Ashton Anderson, *Chessformer: A Unified Architecture for Chess Modeling*, ICLR 2026, arXiv:2605.19091; and Albert W. Hamood, *DSC: Human Chess Move Prediction Models*, ChessDS, 2026, [chessds.com](https://chessds.com/).

## Limitations

- Evaluated on classical games at 2400+; not evaluated below 2400 or on other time controls.
- The raw WDL head is an auxiliary training output. Do not rely on it as a standalone game-result forecast; in our 23M tests it was less accurate than dedicated result models.
- Standalone CPU checks passed on macOS arm64, Windows x64 and Linux x64; Intel Mac passed under Rosetta. Physical Intel Mac, consumer-machine GUI launch, Linux arm64, Python 3.10 and package inference on CUDA/MPS remain untested.
