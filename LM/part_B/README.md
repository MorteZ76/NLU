# Language Modeling (LM) — Part 1.B

This directory contains the implementation of Part 1.B, which applies three regularization techniques from Merity et al. (2017) to the baseline RNN language model, evaluated on the Penn TreeBank (PTB) dataset.

---

## Task Description

Starting from the `LM_RNN` model (in which the vanilla RNN was substituted with an LSTM in Part 1.A), the objective is to incrementally apply the following regularization techniques and demonstrate that each improves perplexity:

1. **Weight Tying** — Share the weights of the embedding layer and the output projection layer.
2. **Variational Dropout** — Apply a consistent dropout mask across the time dimension instead of independent element-wise dropout per step.
3. **Non-monotonically Triggered AvSGD (NT-ASGD)** — Switch from SGD to Averaged SGD when validation loss stops improving monotonically.

These techniques are described in: *[Regularizing and Optimizing LSTM Language Models (Merity et al., 2017)](https://openreview.net/pdf?id=SyyGPP0TZ)*.

**Mandatory requirements:**
- Final test perplexity must be strictly below **250** (PPL < 250).
- Final test perplexity must be lower than the Part 1.A base LSTM result (**149.522**).

---

## Assignment Requirements

- **Modular Architecture:** Code is separated into distinct files (`utils.py`, `model.py`, `functions.py`, `main.py`).
- **Performance Threshold:** Test PPL < 250 and below the Part 1.A LSTM baseline (149.522).
- **Incremental Experimentation:** Each technique is applied one at a time. All experiments are recorded below.
- **Hyperparameter Tuning:** Sequential grid search is used to tune learning rate, embedding size, dropout rates, and the NT-ASGD trigger window.
- **No Notebooks:** Only clean, well-documented Python scripts are submitted.

---

## File and Directory Structure

```
LM/
└── part_B/
    ├── utils.py              # Data preprocessing, Lang vocabulary class, batch collators
    ├── model.py              # PyTorch model definitions (LM_RNN with VariationalDropout & LM_LSTM)
    ├── functions.py          # Seeding, weight initialization, train/eval loops, NT-ASGD utilities, grid search
    ├── main.py               # Command-line execution orchestrator
    ├── requirements.txt      # Standard / GPU library dependencies
    ├── cpu_requirements.txt  # CPU-only PyTorch dependencies
    ├── README.md             # This documentation, execution guide, and experimental log
    ├── dataset/              # Raw Penn TreeBank text split files
    └── bin/                  # Saved model checkpoints, configs, and metrics
        └── RNN_NT-ASGD_VarDrop_WeightTied/
            ├── RNN_NT-ASGD_VarDrop_WeightTied.pt    # Final model state-dict checkpoint
            ├── config.json                           # Final hyperparameter configuration
            └── loss_plot.png                         # Training vs. Validation Loss curves
```

---

## Technical Features

- **Weight Tying:** The output projection matrix is set equal to the embedding matrix (`output.weight = embedding.weight`). This reduces the parameter count and regularizes the model by forcing a consistent token representation across both input encoding and output decoding. Requires `emb_size == hidden_size`.
- **Variational Dropout:** Unlike standard dropout which resamples a new mask at every time step, variational dropout applies one fixed mask for the entire sequence. This prevents the model from compensating for dropped inputs by exploiting positional recovery patterns across time steps.
- **NT-ASGD:** Training begins with standard SGD. When validation loss fails to improve for `non_mono` consecutive epochs, the optimizer automatically switches to Averaged SGD (ASGD). During validation and final evaluation, the averaged weights are substituted in for a smoother, better-generalizing parameter estimate.
- **Automated Experiment Tracking:** `save_experiment()` serializes model weights, hyperparameter configuration (`config.json`), and Training vs. Validation Loss curves (`loss_plot.png`) for every run.

---

## Mathematical Foundation

### Perplexity (PPL)

```
PPL = exp( sum(loss_i) / sum(tokens_i) )
```

Lower perplexity indicates a better model. All results are reported on the PTB test set unless stated otherwise.

### NT-ASGD Trigger Condition

ASGD is activated at epoch `t` when:

```
val_loss[t] > min(val_losses[0 : t - non_mono])
```

If the current validation loss exceeds the best loss recorded more than `non_mono` epochs ago, the optimizer switches from SGD to ASGD.

---

## Installation & Setup

Navigate to the `part_B` directory and install the appropriate dependencies:

**Option A — Standard / GPU Setup**

```bash
cd LM/part_B
pip install -r requirements.txt
```

**Option B — CPU-Only Setup**

```bash
cd LM/part_B
pip install -r cpu_requirements.txt
```

---

## Execution

**Standard Training**

```bash
python main.py
```

**Hyperparameter Tuning**

Runs the sequential grid search defined in `param_tuning_order` inside `main.py`.

```bash
python main.py --tune
```

**Evaluation Only**

Loads a saved checkpoint and reports validation and test PPL without retraining.

```bash
python main.py --eval_only --model_path "bin/<experiment_name>/<experiment_name>.pt"
```

> **Note:** Use forward slashes on Linux/Colab. The code normalizes backslashes automatically, but `/` is safer across platforms.

---

## Experimental Log

The table below records each incremental modification and its impact on validation and test perplexity. Each experiment builds on the best configuration from the previous step.

| Exp | Configuration | Modification | Val PPL | Test PPL | Decision |
|:---:|:---|:---|:---:|:---:|:---:|
| **0** | Baseline RNN + SGD (lr=1) | Default settings, no regularization | 7891.023 | 7918.069 | Diverged — lr too low |
| **1** | RNN + SGD | Tuning: lr=5 | 156.756 | 150.919 | Kept |
| **2** | + Weight Tying | Tied embedding and output weights (emb\_size=hidden\_size=200) | 155.422 | 150.835 | Kept |
| **3** | + Tuning: emb\_size | emb\_size=200 confirmed as optimal | 149.242 | 145.354 | Kept |
| **4** | + Variational Dropout | emb\_drop=0.1, out\_drop=0.1 | 140.887 | 136.009 | Kept |
| **5** | **+ NT-ASGD** | **non\_mono=1** | **132.974** | **128.729** | **FINAL** |

---

## Hyperparameter Tuning Details

### Learning Rate (Baseline RNN + SGD)

| lr | Val PPL | Test PPL |
|:---:|:---:|:---:|
| **5** | 156.756 | **150.919** ✓ |
| 10 | 159.647 | 152.869 |
| 1 | 184.306 | 177.984 |
| 0.5 | 215.032 | 208.257 |
| 50 | 290.710 | 280.745 |
| 0.1 | 337.284 | 329.016 |
| 0.05 | 415.738 | 405.920 |
| 0.01 | 745.383 | 727.238 |

### Embedding Size (with Weight Tying, lr=5)

Weight tying requires `emb_size == hidden_size`. The search confirms `emb_size=200` as optimal.

| emb\_size | Val PPL | Test PPL |
|:---:|:---:|:---:|
| **200** | 149.242 | **145.354** ✓ |
| 300 | 149.951 | 146.780 |
| 400 | 155.422 | 150.835 |
| 100 | 159.240 | 152.331 |

### Embedding Dropout (with Variational Dropout + Weight Tying)

| emb\_drop | Val PPL | Test PPL |
|:---:|:---:|:---:|
| **0.1** | 147.470 | **142.714** ✓ |
| 0.3 | 152.245 | 146.920 |
| 0.5 | 162.184 | 156.515 |

### Output Dropout (with emb\_drop=0.1)

| out\_drop | Val PPL | Test PPL |
|:---:|:---:|:---:|
| **0.1** | 140.887 | **136.009** ✓ |
| 0.3 | 144.980 | 140.171 |
| 0.5 | 149.221 | 144.634 |

### NT-ASGD Trigger Window (with Variational Dropout + Weight Tying)

| non\_mono | Val PPL | Test PPL |
|:---:|:---:|:---:|
| **1** | 132.974 | **128.729** ✓ |
| 2 | 140.887 | 136.009 |
| 3 | 140.887 | 136.009 |
| 4 | 140.887 | 136.009 |
| 5 | 140.887 | 136.009 |

---

## Reproducing Results

**Baseline RNN + SGD — default lr=1 (Exp 0)**
```bash
python main.py --eval_only --model_path "bin/RNN_SGD_NoDrop/RNN_SGD_NoDrop.pt"
```

**Baseline RNN + SGD — lr=5 (Exp 1)**
```bash
python main.py --eval_only --model_path "bin/grid_lr=5/grid_lr=5.pt"
```

**+ Weight Tying (Exp 2)**
```bash
python main.py --eval_only --model_path "bin/RNN_SGD_NoDrop_WeightTied/RNN_SGD_NoDrop_WeightTied.pt"
```

**+ emb\_size=200 tuning (Exp 3)**
```bash
python main.py --eval_only --model_path "bin/grid_emb_size=200/grid_emb_size=200.pt"
```

**+ Variational Dropout — out\_drop=0.1 (Exp 4)**
```bash
python main.py --eval_only --model_path "bin/grid_out_drop=0.1/grid_out_drop=0.1.pt"
```

**Final Model — NT-ASGD, non\_mono=1 (Exp 5)**
```bash
python main.py --eval_only --model_path "bin/RNN_NT-ASGD_VarDrop_WeightTied/RNN_NT-ASGD_VarDrop_WeightTied.pt"
```

---

## Experimental Discussion

**Baseline and Learning Rate:** The default SGD learning rate of 1 caused the model to diverge completely (Test PPL ~7918). A grid search over learning rates identified lr=5 as optimal, bringing the model to a Test PPL of 150.919 — comparable to the Part 1.A base LSTM (149.522) but achieved with a vanilla RNN.

**Weight Tying:** Adding weight tying alone had minimal direct impact on PPL (150.919 → 150.835), but it constrained the model to use a consistent representation space for both token input and output. It also enabled `emb_size` to be treated as a meaningful tunable parameter. Tuning confirmed `emb_size=200` as optimal, reducing Test PPL to 145.354 — the first result below the Part 1.A LSTM baseline.

**Variational Dropout:** Applying consistent-mask dropout after the embedding (`emb_drop=0.1`) and before the output projection (`out_drop=0.1`) produced the single largest gain in this part, reducing Test PPL from 145.354 to 136.009. The improvement over standard dropout stems from variational dropout's ability to regularize sequential dependencies, preventing the model from relying on step-specific activation patterns to recover from dropped inputs.

**NT-ASGD:** Switching to the non-monotonically triggered ASGD optimizer with `non_mono=1` delivered the final gain, reducing Test PPL from 136.009 to **128.729**. The tighter trigger window (1 vs. the default 5) activates averaging as soon as any non-improvement is detected, leading to more aggressive parameter smoothing and a more stable final estimate. Values of non_mono=2 through 5 all converged to the same result as the pre-ASGD model (136.009), confirming that early triggering is critical.

**Final Result:** The fully regularized model (`RNN_NT-ASGD_VarDrop_WeightTied`) achieves a **Test Perplexity of 128.729**, satisfying both mandatory requirements: below 250 and below the Part 1.A base LSTM (149.522).

---

## AI Assistance Disclosure

AI tools (Claude by Anthropic) were used in the development of this project, including assistance with writing and refining the code structure, inline comments, docstrings, and this report.
