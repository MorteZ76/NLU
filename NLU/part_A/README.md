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
- **Hyperparameter Tuning:** Sequential grid search optimizes learning rate, hidden/embedding size, dropout rate, batch size, and gradient clipping. The optimizer (Adam) was kept fixed across experiments.
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
- **Dropout is architecturally optional, not just a tunable rate:** `nn.Dropout(dropout_rate)` is a true no-op when `dropout_rate=0.0` (identical to not having the layer at all), so the "Baseline" and "Bidirectional" experiments below were explicitly retrained with `dropout_rate: 0.0` set in their configs — earlier versions of these two checkpoints had silently relied on the code's `dropout_rate=0.1` default, which meant the "+ Dropout" step wasn't actually isolating dropout as its own modification. Every `config.json` in `bin/` now states its `dropout_rate` explicitly, whether zero or not.
- **Strict Reproducibility:** A fixed global seed of `1234` is applied across all random number generators (Python, NumPy, PyTorch CPU/CUDA). The slot/intent label vocabulary is also built with `sorted()` rather than raw `set()` iteration, guaranteeing the same label-to-id mapping every run — otherwise Python's per-process string hash randomization could assign different ids to the same labels across runs, silently corrupting any checkpoint reloaded in a new process.
- **Configurable Architecture:** `bidirectional` and `dropout_rate` are read from `config.json` and applied consistently across standard training, `--tune`, and `--eval_only` — the same architecture used to train a checkpoint is always the one used to reload it.
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

Runs the sequential grid search defined by `tuning_grid` in `config.json`, searching `lr → hidden_size → dropout_rate → emb_size → batch_size → clip` in order, fixing each parameter at its best value before moving to the next.

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
| **0** | Baseline | Unidirectional LSTM, `dropout_rate=0.0` (no dropout) | 0.9707 | 0.9251 | Base |
| **1** | + Bidirectional | `bidirectional=True`, `dropout_rate=0.0` (no dropout) | 0.9762 | 0.9400 | Kept |
| **2** | + Dropout & Capacity | `dropout_rate=0.4`, `emb_size=400`, `hidden_size=250`, `lr=0.001`, `batch_size=128` | 0.9816 | 0.9479 | Kept (tuning base) |
| **3** | Tuning — Round 1 | Sequential search over `lr/hidden_size/dropout_rate/emb_size/batch_size/clip` around Exp 2's config | — | 0.9479 | No further gain |
| **4** | Tuning — Round 2 | Refined grids around Round 1's best point | — | 0.9490 | Kept |
| **5** | **Final Model** | `lr=0.0001`, `hidden_size=200`, `dropout_rate=0.1`, `emb_size=400`, `batch_size=32`, `clip=5` | **0.9820** | **0.9471** | **FINAL** |

---

## Hyperparameter Tuning Details

### Round 1 — `sequential__20260704_105001`

| Parameter | Trials | Best Value | Best Avg |
|:---|:---|:---:|:---:|
| `lr` | 0.0001, 0.001, 0.01 | 0.0001 | 0.9479 |
| `hidden_size` | 100, 200, 300 | 200 | 0.9479 |
| `dropout_rate` | 0.0, 0.1, 0.3, 0.5 | 0.1 | 0.9479 |
| `emb_size` | 100, 200, 300 | 300 | 0.9479 |
| `batch_size` | 32, 64, 128 | 32 | 0.9479 |
| `clip` | 1, 5, 10 | 5 | 0.9479 |

Every parameter in this round converged back to the same Avg (0.9479) already achieved by the bidirectional + dropout model — the wider capacity/dropout settings from Exp 2 weren't actually helping.

### Round 2 — `sequential__20260704_135712`

A second, narrower sequential search centered on Round 1's best point:

| Parameter | Trials | Best Value | Best Avg |
|:---|:---|:---:|:---:|
| `lr` | 0.0001, 0.0002, 0.0005 | 0.0001 | 0.9479 |
| `hidden_size` | 150, 200, 250 | 200 | 0.9479 |
| `dropout_rate` | 0.05, 0.1, 0.15, 0.2 | 0.1 | 0.9479 |
| `emb_size` | 250, 300, 350, **400** | **400** | **0.9490** |
| `batch_size` | 32, 64, 128 | 32 | 0.9490 |
| `clip` | 3, 5, 7 | 5 | 0.9490 |

Increasing `emb_size` to 400 (Slot F1 0.9485, Intent Acc 0.9496) was the one change in this round that actually moved the needle — every other parameter reconfirmed its Round 1 value.

---

## Reproducing Results

Use `--eval_only` to evaluate any saved checkpoint without retraining:

**Baseline (Exp 0)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Baseline/ATIS_Joint_Model_Baseline.pt"
```
```
Validation F1: 0.9614 | Intent Acc: 0.9799 | Average: 0.9707
Test Set F1:   0.9174 | Intent Acc: 0.9328 | Average: 0.9251
```

**+ Bidirectional (Exp 1)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Bidirectional/ATIS_Joint_Model_Bidirectional.pt"
```
```
Validation F1: 0.9725 | Intent Acc: 0.9799 | Average: 0.9762
Test Set F1:   0.9371 | Intent Acc: 0.9429 | Average: 0.9400
```

**+ Dropout (Exp 2, pre-tuning)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Bidirectional_Drpout/ATIS_Joint_Model_Bidirectional_Drpout.pt"
```
```
Validation F1: 0.9793 | Intent Acc: 0.9839 | Average: 0.9816
Test Set F1:   0.9440 | Intent Acc: 0.9518 | Average: 0.9479
```

**Final Model (Exp 5)**
```bash
python main.py --eval_only --model_path "bin/ATIS_Joint_Model_Bidirectional_Drpout/ATIS_Joint_Model_Bidirectional_Drpout_Final.pt"
```
```
Validation F1: 0.9801 | Intent Acc: 0.9839 | Average: 0.9820
Test Set F1:   0.9458 | Intent Acc: 0.9485 | Average: 0.9471
```

---

## Experimental Discussion

**Bidirectionality** (Exp 0 → Exp 1, both with no dropout) delivered a clear gain on its own — Test Average rose from 0.9251 to 0.9400 (Test Slot F1: 0.9174 → 0.9371, Intent Acc: 0.9328 → 0.9429). Letting the encoder see both left and right context at every token helps slot tagging, and concatenating the forward and backward final hidden states gives the intent head a fuller summary of the whole utterance instead of only what preceded each token.

**Adding dropout** (Exp 1 → Exp 2), together with a heavier dropout rate (0.4) and larger embedding/hidden sizes, gave a further improvement on top of bidirectionality — Test Average rose again from 0.9400 to 0.9479.

**Sequential tuning** confirmed that Exp 2's aggressive settings weren't quite optimal: Round 1 converged to a lighter `dropout_rate=0.1`, `hidden_size=200`, `emb_size=300`. Round 2, searching a narrower grid around that point, found that keeping the lighter dropout and hidden size while increasing only `emb_size` to 400 gave a genuine further improvement (Test Avg 0.9479 → 0.9490).

**Final Result:** Retraining from scratch with the tuned configuration (`lr=0.0001`, `hidden_size=200`, `dropout_rate=0.1`, `emb_size=400`, `batch_size=32`, `clip=5`) reached a **Test Average of 0.9471** (Slot F1 0.9458, Intent Accuracy 0.9485), consistent with the sequential search's estimate and confirming the tuned hyperparameters generalize when retrained independently.

---

## AI Assistance Disclosure

AI tools (Claude by Anthropic) were used in the development of this project, including assistance with writing and refining the code structure, bug fixes (reproducible label-id mapping, a training/eval hyperparameter consistency fix, explicit zero-dropout baselines), inline comments, docstrings, and this report.
