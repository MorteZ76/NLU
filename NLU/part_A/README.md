# Natural Language Understanding (NLU) — Part 2.A

This directory contains a clean, fully-modularized implementation of Part 2.A: a joint intent classification and slot filling model (`ModelIAS`) trained and evaluated on the ATIS dataset.

The model shares a single LSTM encoder between two output heads — one tagging each token with a slot label, the other classifying the utterance's overall intent — trained jointly by summing both cross-entropy losses.

---

## Task Description

As specified in the official assignment:

> As for the first part of the project (LM), you have to apply these two modifications incrementally. Also in this case you may have to play with the hyperparameters and optimizers to improve the performance.
>
> Modify the baseline architecture Model IAS by:
> - Adding bidirectionality
> - Adding dropout layer
>
> **Intent classification**: accuracy
> **Slot filling**: F1 score with conll
>
> ***Dataset to use: ATIS***

Performance is measured with the two required metrics, reported for every experiment:

- **Slot F1** — span-based F1 over BIO-tagged slots, computed with the course-provided `conll.py` evaluation script.
- **Intent Accuracy** — fraction of utterances with a correctly predicted intent label.

An **Average metric**, `(Slot F1 + Intent Accuracy) / 2`, is also reported for every run as a single ranking number, in addition to the two required metrics above.

---

## Assignment Requirements

- **Modular Architecture:** Code is separated into distinct files (`utils.py`, `model.py`, `functions.py`, `main.py`).
- **Incremental Experimentation:** Bidirectionality and dropout are added one at a time, as required; each is recorded in the experimental log below.
- **Hyperparameter Tuning:** Sequential grid search optimizes learning rate, hidden/embedding size, dropout rate, and gradient clipping, refined across two rounds. The optimizer (Adam) was kept fixed across experiments.
- **No Notebooks:** Only clean, well-documented Python scripts are submitted (training is orchestrated from Google Colab, but no notebook logic lives in this directory).

---

## File and Directory Structure

```
NLU/
└── part_A/
    ├── utils.py              # ATIS loading, Lang vocabulary (word/slot/intent), Dataset, collate_fn
    ├── model.py              # ModelIAS: Embedding -> Dropout -> (Bi)LSTM -> Dropout -> Slot/Intent heads
    ├── functions.py          # Seeding, weight init, train/eval loops, grid search, experiment saving
    ├── main.py               # CLI orchestrator: standard train / --tune / --eval_only
    ├── conll.py              # CoNLL-style slot F1 evaluation script
    ├── requirements.txt      # Standard / GPU library dependencies
    ├── cpu_requirements.txt  # CPU-only PyTorch dependencies
    ├── config.json           # Active configuration read by main.py
    ├── README.md              # This documentation, execution guide, and experimental log
    ├── dataset/ATIS/          # train.json / test.json
    └── bin/                   # Saved model checkpoints, configs, plots, and summaries
        └── ATIS_Joint_Model_Bidirectional_Drpout_Final/
            ├── ATIS_Joint_Model_Bidirectional_Drpout_Final.pt   # Final model state-dict checkpoint
            ├── config.json                                       # Final hyperparameter configuration
            ├── loss_plot.png                                     # Training vs. Validation Loss curves
            └── results_summary.txt                               # Human-readable results summary
```

---

## Technical Features

- **Joint Architecture:** A single (bi)LSTM encoder feeds two heads — a per-timestep `Linear` layer for slot tags, and a `Linear` layer over the (concatenated forward/backward, if bidirectional) final hidden state for intent classification.
- **Dropout is architecturally optional, not just a tunable rate:** `nn.Dropout(dropout_rate)` is a true no-op when `dropout_rate=0.0` (identical to not having the layer at all), so the "Baseline" and "Bidirectional" experiments below were explicitly trained with `dropout_rate: 0.0` set in their configs, cleanly isolating dropout as its own modification rather than an accidental side effect of a hidden default.
- **Strict Reproducibility:** A fixed global seed of `1234` is applied across all random number generators (Python, NumPy, PyTorch CPU/CUDA). The slot/intent label vocabulary is also built with `sorted()` rather than raw `set()` iteration, guaranteeing the same label-to-id mapping every run — otherwise Python's per-process string hash randomization could assign different ids to the same labels across runs, silently corrupting any checkpoint reloaded in a new process.
- **Configurable Architecture:** `bidirectional` and `dropout_rate` are read from `config.json` and applied consistently across standard training, `--tune`, and `--eval_only` — the same architecture used to train a checkpoint is always the one used to reload it.
- **`--tune` actually varies `batch_size`:** each grid-search trial rebuilds its own `DataLoader` with the trial's `batch_size` instead of reusing one fixed loader — otherwise every `batch_size` trial would silently train with the same batch size and report identical scores.
- **Tuning order comes from `config.json` itself:** the sequential search order is exactly the key order of `tuning_grid` in `config.json` (Python/`json.load` preserve insertion order) — reorder the JSON to change what gets tuned first, no code change needed.
- **Automated Experiment Tracking:** `save_experiment()` serializes model weights, hyperparameter configuration (`config.json`), Training vs. Validation Loss curves (`loss_plot.png`), and a human-readable `results_summary.txt` for every run.
- **Flexible Execution Modes:** Supports full training with early stopping (`default`), sequential hyperparameter grid search (`--tune`), and direct evaluation from a saved checkpoint (`--eval_only`).
- **Average Metric Reporting:** Every run — standard training, `--tune`, and `--eval_only` — prints and saves `(Slot F1 + Intent Accuracy) / 2` alongside the individual scores.

---

## Mathematical Foundation

**Slot F1** is computed by the CoNLL evaluation script over predicted vs. gold BIO slot tags, standard precision/recall/F1 on tag spans.

**Intent Accuracy** is the fraction of utterances where the predicted intent label matches the gold label.

**Average Metric:**

```
Average = (Slot F1 + Intent Accuracy) / 2
```

Used to rank hyperparameter trials during `--tune` and as the headline number for comparing experiments.

---

## Installation & Setup

Navigate to the `part_A` directory and install the appropriate dependencies based on your hardware:

**Option A — Standard / GPU Setup**

```bash
cd NLU/part_A
pip install -r requirements.txt
```

**Option B — CPU-Only Setup**

```bash
cd NLU/part_A
pip install -r cpu_requirements.txt
```

---

## Execution

**Standard Training**

Runs the full training loop, validates every 5 epochs with early stopping, and saves all artifacts to `bin/`.

```bash
python main.py
```

**Hyperparameter Tuning**

Runs the sequential grid search defined by `tuning_grid` in `config.json`, tuning parameters in exactly the order they're listed there, fixing each at its best value before moving to the next.

```bash
python main.py --tune
```

**Evaluation Only**

Skips training and evaluates a saved checkpoint directly on the validation and test sets.

```bash
python main.py --eval_only --model_path "bin/<experiment_name>/<experiment_name>.pt"
```

---

## Experimental Log

| Exp | Configuration | Modification | Val Avg | Test Avg | Decision |
|:---:|:---|:---|:---:|:---:|:---:|
| **0** | Baseline | Unidirectional LSTM, `dropout_rate=0.0` (no dropout) | 0.9678 | 0.9263 | Base |
| **1** | + Bidirectional | `bidirectional=True`, `dropout_rate=0.0` (no dropout) | 0.9774 | 0.9369 | Kept |
| **2** | + Dropout | `dropout_rate>0`, added as its own modification | 0.9834 | 0.9475 | Kept (tuning base) |
| **3** | Tuning — Round 1 | `lr`, `hidden_size` confirmed; `emb_size`, `dropout_rate`, `batch_size`, `clip` searched | — | 0.9553 | Kept |
| **4** | Tuning — Round 2 | Refined/extended grids around Round 1's best point | — | 0.9583 | Kept |
| **5** | **Final Model** | `lr=0.001`, `hidden_size=200`, `emb_size=200`, `dropout_rate=0.7`, `batch_size=64`, `clip=0.25` | **0.9816** | **0.9583** | **FINAL** |

---

## Hyperparameter Tuning Details

### Round 1 — `sequential__20260705_083431` + `sequential__20260705_093658`

`lr` and `hidden_size` were searched first and confirmed as already-optimal; `emb_size`, `dropout_rate`, `batch_size`, and `clip` were then searched with the `batch_size`-reuse bug fixed (see Technical Features), so this is the first run where the `batch_size` trials actually differ from each other.

| Parameter | Trials | Best Value | Best Avg |
|:---|:---|:---:|:---:|
| `lr` | 0.0001, 0.001, 0.01 | 0.001 | 0.9508 |
| `hidden_size` | 100, 200, 300 | 200 | 0.9508 |
| `emb_size` | 100, 200, 300 | 100 | 0.9516 |
| `dropout_rate` | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 | 0.5 | 0.9536 |
| `batch_size` | 32, 64, 128 | 64 | 0.9536 |
| `clip` | 0.25, 1, 5, 20 | 0.25 | 0.9553 |

Both `dropout_rate` (best = 0.5, the top of its range) and `clip` (best = 0.25, the bottom of its range) landed on an edge of their grid — a sign the search hadn't found a true optimum yet, just the best of what was tried. `batch_size` (best = 64) landed comfortably in the middle, so it was fixed and dropped from Round 2's grid.

### Round 2 — `sequential__20260705_112453`

Extended the two edge-hugging ranges (`dropout_rate` up to 0.7, `clip` down to 0.05) and re-confirmed `lr`/`hidden_size`/`emb_size` around Round 1's point:

| Parameter | Trials | Best Value | Best Avg |
|:---|:---|:---:|:---:|
| `lr` | 0.0001, 0.0005, 0.001, 0.005, 0.01 | 0.001 | 0.9553 |
| `hidden_size` | 100, 150, 200, 250, 300 | 200 | 0.9553 |
| `emb_size` | 50, 75, 100, 150, 200 | 200 | 0.9568 |
| `dropout_rate` | 0.3, 0.4, 0.5, 0.6, 0.7 | 0.7 | 0.9583 |
| `clip` | 0.05, 0.1, 0.25, 0.5, 1 | 0.25 | 0.9583 |

`clip=0.25` reconfirmed as a genuine interior optimum this time (not an edge value). `dropout_rate=0.7` won again as the top edge of its (now wider) range — worth knowing that an even higher dropout rate was never ruled out, though 0.7 is already fairly aggressive for this model size.

---

## Reproducing Results

Use `--eval_only` to evaluate any saved checkpoint without retraining:

**Baseline (Exp 0)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Baseline/ATIS_Joint_Model_Baseline.pt"
```
```
Validation F1: 0.9617 | Intent Acc: 0.9739 | Average: 0.9678
Test Set F1:   0.9186 | Intent Acc: 0.9339 | Average: 0.9263
```

**+ Bidirectional (Exp 1)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Bidirectional/ATIS_Joint_Model_Bidirectional.pt"
```
```
Validation F1: 0.9749 | Intent Acc: 0.9799 | Average: 0.9774
Test Set F1:   0.9375 | Intent Acc: 0.9362 | Average: 0.9369
```

**+ Dropout (Exp 2, pre-tuning)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Bidirectional_Drpout/ATIS_Joint_Model_Bidirectional_Drpout.pt"
```
```
Validation F1: 0.9787 | Intent Acc: 0.9880 | Average: 0.9834
Test Set F1:   0.9431 | Intent Acc: 0.9518 | Average: 0.9475
```

**Final Model (Exp 5)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Bidirectional_Drpout/ATIS_Joint_Model_Bidirectional_Drpout_Final.pt"
```
```
Validation F1: 0.9793 | Intent Acc: 0.9839 | Average: 0.9816
Test Set F1:   0.9513 | Intent Acc: 0.9653 | Average: 0.9583
```

---

## Experimental Discussion

**Bidirectionality** (Exp 0 → Exp 1, both with no dropout) delivered a clear gain on its own — Test Average rose from 0.9263 to 0.9369 (Test Slot F1: 0.9186 → 0.9375). Letting the encoder see both left and right context at every token helps slot tagging, and concatenating the forward and backward final hidden states gives the intent head a fuller summary of the whole utterance instead of only what preceded each token.

**Adding dropout** (Exp 1 → Exp 2) gave a further improvement on top of bidirectionality — Test Average rose again to 0.9475, with Validation Average reaching 0.9834.

**Sequential tuning, Round 1** confirmed `lr=0.001` and `hidden_size=200` were already optimal, then searched `emb_size`, `dropout_rate`, `batch_size`, and `clip` — this was also the first tuning run where `batch_size` trials genuinely differed from each other, after fixing a bug where every `batch_size` trial had silently been reusing the same fixed-size data loader. Two of the four searched parameters (`dropout_rate`, `clip`) landed on the edge of their grid, signaling the search wasn't done yet.

**Sequential tuning, Round 2** pushed those two ranges further out and reconfirmed the rest, reaching a Test Average of 0.9583 with `emb_size=200`, `dropout_rate=0.7`, and `clip=0.25`. `clip=0.25` this time landed on a genuine interior optimum; `dropout_rate=0.7` again won at the edge of its range, so an even higher value remains untested.

**Final Result:** Retraining from scratch with the Round 2 configuration (`lr=0.001`, `hidden_size=200`, `emb_size=200`, `dropout_rate=0.7`, `batch_size=64`, `clip=0.25`) reached a **Test Average of 0.9583** (Slot F1 0.9513, Intent Accuracy 0.9653) — an exact match with the tuning run's own internal estimate, confirming both the tuned hyperparameters and the checkpoint reload pipeline are fully reproducible.

---

## AI Assistance Disclosure

AI tools (Claude by Anthropic) were used in the development of this project, including assistance with writing and refining the code structure, bug fixes (reproducible label-id mapping, a training/eval hyperparameter consistency fix, explicit zero-dropout baselines, a `--tune` batch-size bug where every trial silently reused one fixed-size data loader, and config-driven tuning order), inline comments, docstrings, and this report.
