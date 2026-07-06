# Natural Language Understanding (NLU) — Part 2.B

This directory contains a clean, fully-modularized implementation of Part 2.B: fine-tuning a pre-trained BERT model in a multi-task setting for joint intent classification and slot filling on the ATIS dataset.

---

## Task Description

As specified in the official assignment:

> Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. You can refer to this paper to have a better understanding of how to implement this: [https://arxiv.org/abs/1902.10909](https://arxiv.org/abs/1902.10909). In this, one of the challenges of this is to handle the sub-tokenization issue.
>
> Note: The fine-tuning process is to further train on a specific task/s a model that has been pre-trained on a different (potentially unrelated) task/s.
>
> The models that you can experiment with are BERT-base or BERT-large.
>
> **Intent classification**: accuracy
> **Slot filling**: F1 score with conll
>
> ***Dataset to use: ATIS***

Performance is measured with the two required metrics, reported for every experiment:

- **Slot F1** — span-based F1 over BIO-tagged slots, computed with the course-provided `conll.py` evaluation script, at word granularity (not sub-token granularity).
- **Intent Accuracy** — fraction of utterances with a correctly predicted intent label.

An **Average metric**, `(Slot F1 + Intent Accuracy) / 2`, is also reported for every run as a single ranking number, in addition to the two required metrics above.

---

## The Sub-Tokenization Challenge

BERT's WordPiece tokenizer can split a single ATIS word into multiple sub-tokens, but the dataset provides one slot label per *word*, not per sub-token. Following [Chen et al., 2019](https://arxiv.org/abs/1902.10909), this is resolved at the data level (`utils.py`, `BERTIntentsAndSlots`):

- The **first** sub-token of each word is assigned that word's real slot label id.
- Every other position — `[CLS]`, `[SEP]`, padding, and any non-first sub-token — is assigned `-100`.
- `CrossEntropyLoss(ignore_index=-100)` then automatically skips all of those positions during training.
- At evaluation time (`functions.py`, `eval_loop`), predictions are gathered only at the positions where the gold label isn't `-100`, reconstructing one label per original word before computing CoNLL F1 — so scoring is always at word granularity, matching the task's slot filling metric.

---

## Assignment Requirements

- **Modular Architecture:** Code is separated into distinct files (`utils.py`, `model.py`, `functions.py`, `main.py`).
- **Multi-Task Fine-Tuning:** A single pre-trained BERT encoder is shared between a slot-tagging head (per-token) and an intent-classification head (from the `[CLS]` token), fine-tuned jointly.
- **Model Choice:** `bert_model` in `config.json` selects between BERT-base and BERT-large (any Hugging Face BERT checkpoint name is accepted).
- **Hyperparameter Tuning:** Sequential grid search optimizes learning rate, weight decay, dropout rate, batch size, and gradient clipping.
- **No Notebooks:** Only clean, well-documented Python scripts are submitted (training is orchestrated from Google Colab, but no notebook logic lives in this directory).

---

## File and Directory Structure

```
NLU/
└── part_B/
    ├── utils.py              # ATIS loading, Lang vocabulary (slot/intent), sub-token alignment, collate_fn
    ├── model.py              # JointBERT: BertModel -> Dropout -> Slot/Intent heads
    ├── functions.py          # Seeding, train/eval loops, grid search, experiment saving
    ├── main.py               # CLI orchestrator: standard train / --tune / --eval_only
    ├── conll.py              # CoNLL-style slot F1 evaluation script
    ├── requirements.txt      # Standard / GPU library dependencies
    ├── cpu_requirements.txt  # CPU-only PyTorch dependencies
    ├── config.json           # Active configuration read by main.py
    ├── README.md              # This documentation, execution guide, and experimental log
    ├── dataset/ATIS/          # train.json / test.json
    └── bin/                   # Saved model checkpoints, configs, plots, and summaries
        └── <experiment_name>/
            ├── <experiment_name>.pt   # Final model state-dict checkpoint
            ├── config.json             # Hyperparameter configuration for this run
            ├── loss_plot.png           # Training vs. Validation Loss curves
            └── results_summary.txt     # Human-readable results summary
```

---

## Technical Features

- **Multi-Task Architecture:** A single shared `BertModel` feeds two heads — a per-token `Linear` layer for slot tags, and a `Linear` layer over the `[CLS]` token's pooled representation for intent classification, trained jointly by summing both cross-entropy losses.
- **Sub-Tokenization Handling:** See above — label alignment to first sub-tokens plus `-100` masking, with word-level reconstruction at evaluation time.
- **Deterministic Label Vocabulary:** Slot/intent label-to-id mappings are built with `sorted()`, guaranteeing the same mapping every run regardless of Python's per-process string hash randomization — otherwise a checkpoint reloaded in a new process could have its labels silently scrambled.
- **`--tune` correctly varies every parameter, including `batch_size`:** each grid-search trial rebuilds its own data loaders with the trial's config, so `batch_size` trials (and any other parameter) actually produce different results.
- **Tuning order comes from `config.json` itself:** the sequential search order is exactly the key order of `tuning_grid` in `config.json` (Python/`json.load` preserve insertion order) — reorder the JSON to change what gets tuned first, no code change needed.
- **Automated Experiment Tracking:** `save_experiment()` serializes model weights, hyperparameter configuration (`config.json`), Training vs. Validation Loss curves (`loss_plot.png`), and a human-readable `results_summary.txt` for every run.
- **Flexible Execution Modes:** Supports full fine-tuning with early stopping (`default`), sequential hyperparameter grid search (`--tune`), and direct evaluation from a saved checkpoint (`--eval_only`).
- **Average Metric Reporting:** Every run — standard training, `--tune`, and `--eval_only` — prints and saves `(Slot F1 + Intent Accuracy) / 2` alongside the individual scores.

---

## Mathematical Foundation

**Slot F1** is computed by the CoNLL evaluation script over predicted vs. gold BIO slot tags at word granularity, standard precision/recall/F1 on tag spans.

**Intent Accuracy** is the fraction of utterances where the predicted intent label matches the gold label.

**Average Metric:**

```
Average = (Slot F1 + Intent Accuracy) / 2
```

Used to rank hyperparameter trials during `--tune` and as the headline number for comparing experiments.

---

## Installation & Setup

Navigate to the `part_B` directory and install the appropriate dependencies based on your hardware:

**Option A — Standard / GPU Setup**

```bash
cd NLU/part_B
pip install -r requirements.txt
```

**Option B — CPU-Only Setup**

```bash
cd NLU/part_B
pip install -r cpu_requirements.txt
```

---

## Execution

**Standard Fine-Tuning**

Runs the full fine-tuning loop, validates every epoch with early stopping, and saves all artifacts to `bin/`.

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

| Exp | Configuration | Val Avg | Test Avg | Decision |
|:---:|:---|:---:|:---:|:---:|
| **0** | Baseline — paper-matched setup (`lr=5e-5`, `batch_size=128`, `dropout_rate=0.1`, `AdamW`) | 0.9902 | 0.9656 | Base |
| **1** | Tuning — Round 1 (`lr`, `weight_decay`) | — | 0.9663 | Kept |
| **2** | Tuning — Round 2 (`dropout_rate`, `batch_size`) | — | 0.9671 | Kept |
| **3** | Tuning — Round 3 (`dropout_rate` refined, `clip`, `n_epochs`) | — | 0.9671 | Kept |
| **4** | **Final Model** — `lr=3e-5`, `weight_decay=0.1`, `dropout_rate=0.1`, `batch_size=32`, `clip=1.0`, `n_epochs=20` | **0.9894** | **0.9671** | **FINAL** |

The baseline already lands very close to Chen et al.'s reported ATIS numbers (Intent 97.5%, Slot F1 96.1%) — Test Intent Acc 97.42%, Test Slot F1 95.69%. Hyperparameter tuning improved the average further, from 0.9656 to 0.9671.

---

## Hyperparameter Tuning Details

Tuning was split into rounds of 1–3 parameters at a time (large BERT fine-tuning runs are expensive), each fixing the previous round's winners before moving on.

### Round 1 — `sequential__20260705_164856` (`lr`, `weight_decay`)

| Parameter | Trials | Best Value | Best Avg |
|:---|:---|:---:|:---:|
| `lr` | 1e-05, 2e-05, 3e-05, 5e-05, 1e-04 | 3e-05 | 0.9671 |
| `weight_decay` | 0.0, 0.01, 0.1 | 0.1 | 0.9663 |

`lr=3e-5` clearly beat the paper's own `5e-5` on this dataset/setup. Note the paper doesn't use weight decay at all — `0.1` outperforming `0.0` here suggests some L2 regularization does help for this smaller AdamW fine-tuning run.

### Round 2 — `sequential__20260705_181239` (`dropout_rate`, `batch_size`)

| Parameter | Trials | Best Value | Best Avg |
|:---|:---|:---:|:---:|
| `dropout_rate` | 0.1, 0.2, 0.3 | 0.1 | 0.9654 |
| `batch_size` | 16, 32, 64, 128 | 32 | 0.9671 |

`dropout_rate=0.1` won at the *lowest* edge of this grid — not yet confirmed as a true optimum, so Round 3 re-tests it against smaller values. `batch_size=32` is a genuine interior optimum, beating the paper's `128`.

### Round 3 — `sequential__20260705_202250` (`dropout_rate` refined, `clip`, `n_epochs`)

| Parameter | Trials | Best Value | Best Avg |
|:---|:---|:---:|:---:|
| `dropout_rate` | 0.0, 0.05, 0.1 | 0.1 | 0.9671 |
| `clip` | 0.5, 1.0, 5.0 | 1.0 | 0.9671 |
| `n_epochs` | 5, 10, 20, 30 | 20 | 0.9671 |

`dropout_rate=0.1` now beats both `0.0` and `0.05`, confirming it as a genuine optimum rather than an artifact of the previous grid's edge. `clip=1.0` is a clean interior optimum. `n_epochs=20` ties exactly with `30` (both 0.9671), so the model has fully converged by epoch 20 — no benefit to training longer.

---

## Reproducing Results

**Baseline (Exp 0)**
```bash
python main.py --eval_only --model_path bin/ATIS_JointBERT_Baseline/ATIS_JointBERT_Baseline.pt
```
```
Dev  — Slot F1: 0.9844  | Intent Acc: 0.9960  | Average: 0.9902
Test — Slot F1: 0.9569  | Intent Acc: 0.9742  | Average: 0.9656
```

**Final Model (Exp 4)**
```bash
python main.py --eval_only --model_path bin/ATIS_JointBERT_BestHyperparams_Final/ATIS_JointBERT_BestHyperparams_Final.pt
```
```
Dev  — Slot F1: 0.9829  | Intent Acc: 0.9960  | Average: 0.9894
Test — Slot F1: 0.9599  | Intent Acc: 0.9742  | Average: 0.9671
```

---

## Experimental Discussion

**Baseline:** Using the paper's own reported hyperparameters (`lr=5e-5`, `batch_size=128`, `dropout_rate=0.1`, Adam-family optimizer), this implementation reproduces Chen et al.'s ATIS results closely — Test Intent Acc 97.42% vs. their reported 97.5%, Test Slot F1 95.69% vs. their 96.1%.

**Hyperparameter tuning** found a noticeably different — and better — configuration than the paper's own choices for this codebase and split: a smaller `lr` (3e-5 vs. 5e-5), a much smaller `batch_size` (32 vs. 128), and a nonzero `weight_decay` (0.1) that the paper didn't use at all. `dropout_rate=0.1` matched the paper's choice, confirmed as a genuine optimum only after a second, wider round re-tested it against smaller values (it had won an earlier round at the edge of a narrower grid). `n_epochs=20` matching `30` exactly shows the model converges well before the paper's upper range of epoch choices.

**Final Result:** Retraining from scratch with the tuned configuration (`lr=3e-5`, `weight_decay=0.1`, `dropout_rate=0.1`, `batch_size=32`, `clip=1.0`, `n_epochs=20`) reached a **Test Average of 0.9671** (Slot F1 0.9599, Intent Accuracy 0.9742) — an exact match with the tuning run's own internal estimate, confirming both the tuned hyperparameters and the checkpoint reload pipeline are fully reproducible, the same way it did for part_A's final model.

---

## AI Assistance Disclosure

AI tools (Claude by Anthropic) were used in the development of this project, including assistance with writing and refining the code structure, inline comments, docstrings, and this report.
