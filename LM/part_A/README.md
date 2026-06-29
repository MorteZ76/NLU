# Language Modeling (LM) — Part 1.A

This directory contains a clean, fully-modularized implementation of Part 1.A: a Baseline Vanilla Recurrent Neural Network Language Model evaluated on the Penn TreeBank (PTB) dataset.

The model is trained autoregressively to predict the next word token by minimizing cross-entropy loss over shifted input sequences.

---

## Task Description

The objective of Part 1.A is to improve upon the baseline `LM_RNN` model by incrementally applying a set of architectural modifications and optimization techniques. The rules are:

- Modifications must be added **one at a time**. If a modification degrades performance it may be removed, but it must still be reported and discussed.
- Performance is measured using **Perplexity (PPL)** and reported for every experiment.
- Hyperparameter optimization — especially the learning rate — is a required part of the task.

**Mandatory modifications (PPL must remain below 250 after each):**

1. Replace `nn.RNN` with a Long Short-Term Memory (`nn.LSTM`) network.
2. Add two dropout layers: one after the embedding layer, one before the final linear layer.
3. Replace the SGD optimizer with AdamW.

---

## Assignment Requirements

As specified by the Teaching Assistants (TAs), this submission fulfills the following academic criteria:

- **Modular Architecture:** Code is separated into distinct files (`utils.py`, `model.py`, `functions.py`, `main.py`).
- **Performance Threshold:** The final test perplexity must fall strictly below **250**.
- **Incremental Experimentation:** Each architectural modification is applied one at a time. Unsuccessful changes are removed before testing the next. All experiments are recorded below.
- **Hyperparameter Tuning:** Sequential grid search is used to optimize learning rate, embedding/hidden dimensions, batch size, and gradient clipping.
- **No Notebooks:** Only clean, well-documented Python scripts are submitted.

---

## File and Directory Structure

```
LM/
└── part_A/
    ├── utils.py              # Data preprocessing, Lang vocabulary class, batch collators
    ├── model.py              # PyTorch model definitions (LM_RNN & LM_LSTM)
    ├── functions.py          # Seeding, weight initialization, train/eval loops, grid search, plot utilities
    ├── main.py               # Command-line execution orchestrator
    ├── requirements.txt      # Standard / GPU library dependencies
    ├── cpu_requirements.txt  # CPU-only PyTorch dependencies
    ├── README.md             # This documentation, execution guide, and experimental log
    ├── dataset/              # Raw Penn TreeBank text split files
    └── bin/                  # Saved model checkpoints, configs, and metrics
        └── LSTM_Dropout_AdamW_Final/
            ├── LSTM_Dropout_AdamW_Final.pt    # Final model state-dict checkpoint
            ├── config.json                    # Final hyperparameter configuration
            └── loss_plot.png                  # Training vs. Validation Loss curves
```

---

## Technical Features

- **Strict Reproducibility:** A fixed global seed of `1234` is applied across all random number generators (Python, NumPy, PyTorch CPU/CUDA) to guarantee deterministic weight initialization and batch ordering.
- **Custom Weight Initialization:** Applies Xavier Uniform for input-to-hidden projections, Orthogonal Initialization for hidden-to-hidden recurrent weights (mitigates gradient instability), and uniform bounds for linear decoder layers.
- **Automated Experiment Tracking:** `save_experiment()` serializes model weights, hyperparameter configuration (`config.json`), and Training vs. Validation Loss curves (`loss_plot.png`) for every run.
- **Flexible Execution Modes:** Supports full training with early stopping (`default`), sequential hyperparameter grid search (`--tune`), and direct evaluation from a saved checkpoint (`--eval_only`).

---

## Mathematical Foundation

### Perplexity (PPL)

Perplexity is the primary evaluation metric. It represents the exponential of the average cross-entropy loss computed over all non-padded tokens:

```
PPL = exp( sum(loss_i) / sum(tokens_i) )
```

Lower perplexity indicates a better-fitting model — the model assigns higher probability to the correct next token on average.

---

## Installation & Setup

Navigate to the `part_A` directory and install the appropriate dependencies based on your hardware:

**Option A — Standard / GPU Setup**

```bash
cd LM/part_A
pip install -r requirements.txt
```

**Option B — CPU-Only Setup**

```bash
cd LM/part_A
pip install -r cpu_requirements.txt
```

---

## Execution

**Standard Training**

Runs the full training loop, validates performance per epoch with early stopping, and saves all artifacts to `bin/`.

```bash
python main.py
```

**Hyperparameter Tuning**

Runs the sequential grid search defined in `param_tuning_order` inside `main.py`. Edit that list to re-enable any commented-out parameter blocks.

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

The table below records each incremental architectural modification and its impact on validation and test perplexity. Modifications were applied sequentially to isolate their individual effect.

| Exp | Model Architecture | Modification | Val PPL | Test PPL | Decision |
|:---:|:---|:---|:---:|:---:|:---:|
| **0** | Baseline RNN | Vanilla recurrent baseline (LM_RNN) | 169.334 | 161.832 | Base |
| **1** | Baseline LSTM | Replaced RNN with LSTM | 154.677 | 149.522 | Kept |
| **2** | LSTM + Dropout | Added 2 dropout layers (emb + output) | 151.661 | 146.902 | Kept |
| **3** | LSTM + Dropout + AdamW | Replaced SGD with AdamW optimizer | 133.246 | 122.838 | Kept |
| **4** | Tuning: Batch Size | `batch_size=16` | 132.970 | 122.477 | Kept |
| **5** | Tuning: Hidden Size | `hidden_size=400` | 125.861 | 117.881 | Kept |
| **6** | Tuning: Learning Rate | `lr=0.001` (retained from baseline) | 125.861 | 117.881 | Kept |
| **7** | Tuning: Embedding Size | `emb_size=400` | 125.286 | 116.656 | Kept |
| **8** | **Final Model** | `clip=0.1` | **123.429** | **115.732** | **FINAL** |

---

## Reproducing Results

Use `--eval_only` to evaluate any saved checkpoint without retraining:

**Baseline RNN (Exp 0)**
```bash
python main.py --eval_only --model_path "bin/Baseline_RNN/Baseline_RNN.pt"
```

**Baseline LSTM (Exp 1)**
```bash
python main.py --eval_only --model_path "bin/Baseline_LSTM/Baseline_LSTM.pt"
```

**LSTM + Dropout (Exp 2)**
```bash
python main.py --eval_only --model_path "bin/LSTM_with_Dropout/LSTM_with_Dropout.pt"
```

**LSTM + Dropout + AdamW (Exp 3)**
```bash
python main.py --eval_only --model_path "bin/LSTM_Dropout_AdamW/LSTM_Dropout_AdamW.pt"
```

**Final Optimized Model (Exp 8)**
```bash
python main.py --eval_only --model_path "bin/LSTM_Dropout_AdamW_Final/LSTM_Dropout_AdamW_Final.pt"
```

---

## Experimental Discussion

**Architectural Improvements:** Replacing the vanilla RNN with an LSTM delivered the most immediate structural gain — Test PPL dropped from ~161 to ~149 — due to the LSTM's gating mechanism mitigating the vanishing gradient problem. Adding dropout layers provided a modest but consistent regularization benefit.

**Optimizer Impact:** The single largest performance jump came from switching SGD to AdamW. Test PPL fell by ~24 points (146.9 → 122.8), confirming that adaptive learning rate optimization with decoupled weight decay is far better suited to this loss landscape.

**Sequential Grid Search:** After the architecture was finalized, a sequential grid search systematically reduced perplexity further:
- Increasing `hidden_size` and `emb_size` to 400 improved the model's capacity to capture long-range semantic relationships, reducing Test PPL by ~6 points.
- Tightening gradient clipping to `clip=0.1` stabilized AdamW's parameter updates in the final tuning stage.

**Final Result:** The optimized configuration (`LSTM_Dropout_AdamW_Final`) achieves a **Test Perplexity of 115.732**, well below the mandatory threshold of 250.

---

## AI Assistance Disclosure

AI tools (Claude by Anthropic) were used in the development of this project, including assistance with writing and refining the code structure, inline comments, docstrings, and this report.
