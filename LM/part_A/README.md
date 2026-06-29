# **Language Modeling (LM) \- Part 1.A**

This repository contains the clean, fully-modularized implementation of Part 1.A (Baseline Vanilla Recurrent Neural Network Language Model) evaluated on the Penn TreeBank (PTB) dataset.

The baseline model is trained autoregressively to predict the next word token by optimizing cross-entropy loss over shifted inputs.

## **Assignment Requirements & Guidelines**

As instructed by the Teaching Assistants (TAs), this sub-folder is designed to fulfill the following strict operational and academic criteria:

* **Modular Architecture:** The code must be completely separated into distinct files (utils.py, model.py, functions.py, and main.py) inside this subdirectory.  
* **Mandatory Performance Threshold:** Through hyperparameter optimization and incremental modifications, the final achieved test perplexity must be strictly below 250 .  
* **Incremental Experimentation:** Modifications to the baseline architecture must be added one at a time. If a specific technique degrades performance, it can be removed before testing the next modification. Both successful and unsuccessful experiments must be recorded.  
* **Hyperparameter Tuning:** Active optimization of critical parameters (specifically the learning rate, embedding/hidden dimensions, and batch sizes) is required to minimize Perplexity.  
* **No Notebooks:** Only clean, bug-free, and well-documented Python scripts are accepted.

## **File and Directory Structure**

LM/  
└── Part\_A/  
    ├── utils.py              \# Data preprocessors, Lang vocabulary class, and batch collators  
    ├── model.py              \# PyTorch model definition (LM\_RNN & LM\_LSTM)  
    ├── functions.py          \# Seeding, weight initializers, train/eval loops, and plot utilities  
    ├── main.py               \# Command-line-driven execution orchestrator  
    ├── requirements.txt      \# Project library dependencies (Standard/GPU)  
    ├── cpu\_requirements.txt  \# CPU-only PyTorch library dependencies  
    ├── README.md             \# This academic documentation, execution guide, and experimental log  
    ├── dataset/              \# Stored raw Penn TreeBank text split files  
    └── bin/                  \# Saved model checkpoints, config parameters, and metrics  
        └── LSTM\_Dropout\_AdamW\_Final/  
            ├── LSTM\_Dropout\_AdamW\_Final.pt    \# Final PyTorch model state-dict checkpoint  
            ├── config.json                    \# Final recorded hyperparameters  
            └── loss\_plot.png                  \# Training vs. Validation Loss curves

## **Technical Features**

* **Strict Seeding for Reproducibility:** A fixed global seed of 1234 is applied across all random number generators (Python, NumPy, and PyTorch CPU/CUDA backends) to ensure deterministic weights initialization and sequence batching.  
* **Smart Parameter Initializations:** Applies professional initializations (Xavier Uniform for input-to-hidden projections, Orthogonal Initialization for hidden-to-hidden recurrent weights to mitigate gradient instability, and Uniform bounding for linear decoding layers).  
* **Automated Experiment Tracking & Visualization:** The save\_experiment function automatically serializes model weights, saves the exact hyperparameter configuration (config.json), and generates visual plots of the Training vs. Validation Loss curves (loss\_plot.png) for immediate visual evaluation.  
* **Flexible Action Modes:** Supports running a full training loop with early stopping, hyperparameter tuning, or loading pre-trained parameters directly to evaluate performance on the test set.

## **Mathematical Foundations**

### **Perplexity (PPL)**

The core metric evaluated is Perplexity, which mathematically represents the exponential of the average Cross-Entropy loss computed over all non-padded tokens  :

## **Installation & Setup**

Before running the model, make sure to install all the necessary dependencies. Navigate to the Part\_A directory and choose the appropriate requirements file based on your hardware:

**Option A: Standard / GPU Setup**

For environments with CUDA-compatible GPUs:

cd LM/Part\_A  
pip install \-r requirements.txt

**Option B: CPU-Only Setup**

If you do not have a dedicated GPU and wish to install lighter, CPU-only PyTorch binaries:

cd LM/Part\_A  
pip install \-r cpu\_requirements.txt

## **Execution Guidelines**

**Option 1: Standard Training**

Runs the full training corpus, validates performance, and outputs artifacts (checkpoints, configs, and plots) to the /bin directory.

python main.py

**Option 2: Hyperparameter Tuning**

Runs the sequential grid search mapped out in the configuration file.

python main.py \--tune

## **Experimental Log & Results**

Below is the record of incremental architectural modifications, hyperparameter configurations, and their corresponding validation and test Perplexities. Modifications were applied sequentially to map their exact impact on the model.

| Exp ID | Model Architecture & State | Modifications / Tuning Step | Validation PPL | Test PPL | Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **0** | **Baseline RNN** (LM\_RNN) | Vanilla Recurrent Baseline | 169.334 | 161.832 | Base |
| **1** | Baseline LSTM | Replaced RNN with LSTM | 154.677 | 149.522 | Kept |
| **2** | LSTM \+ Dropout | Added 2 Dropout layers | 151.661 | 146.902 | Kept |
| **3** | LSTM \+ Dropout \+ AdamW | Replaced SGD with AdamW | 133.246 | 122.838 | Kept |
| **4** | LSTM Tuning: Batch Size | batch\_size \= 16 | 132.970 | 122.477 | Kept |
| **5** | LSTM Tuning: Hidden Size | hidden\_size \= 400 | 125.861 | 117.881 | Kept |
| **6** | LSTM Tuning: Learning Rate | lr \= 0.001 (Retained Best) | 125.861 | 117.881 | Kept |
| **7** | LSTM Tuning: Embedding Size | emb\_size \= 400 | 125.286 | 116.656 | Kept |
| **8** | **Final Model** | **clip \= 0.1** | **123.429** | **115.732** | **FINAL** |

### **Reproducing the Models**

You can skip training and directly evaluate the performance of any checkpointed model from the table above using the following specific commands:

**Baseline RNN (Exp 0):**

python main.py \--eval\_only \--model\_path "bin/Baseline\_RNN/Baseline\_RNN.pt"

**LSTM Architecture (Exp 1):**

python main.py \--eval\_only \--model\_path "bin/Baseline\_LSTM/Baseline\_LSTM.pt"

**LSTM \+ Dropout (Exp 2):**

python main.py \--eval\_only \--model\_path "bin/LSTM\_with\_Dropout/LSTM\_with\_Dropout.pt"

**LSTM \+ Dropout \+ AdamW (Exp 3):**

python main.py \--eval\_only \--model\_path "bin/LSTM\_Dropout\_AdamW/LSTM\_Dropout\_AdamW.pt"

**Final Optimized Model (Exp 8):**

python main.py \--eval\_only \--model\_path "bin/LSTM\_Dropout\_AdamW\_Final/LSTM\_Dropout\_AdamW\_Final.pt"

### **Experimental Observations & Discussion**

**Architectural Improvements:** The transition from a vanilla RNN to an LSTM immediately granted a massive performance boost (Test PPL dropped from \~161 to \~149), successfully mitigating the vanishing gradient problem inherent in standard RNNs. Adding dropout provided a minor but welcome regularization boost, ensuring the model generalizes better over the PTB corpus.

**Optimizer Impact:** The most significant single leap in performance came from switching the optimizer from standard SGD to AdamW. Test PPL plummeted by an impressive 24 points (146.9 → 122.8), proving that adaptive learning rate optimization with decoupled weight decay is far more effective for navigating this loss landscape.

**Hyperparameter Optimization (Sequential Grid Search):** Following the architectural lockdown, a sequential grid search systematically minimized the perplexity further:

* Increasing the network's capacity (hidden\_size to 400 and emb\_size to 400\) allowed the LSTM to capture much deeper semantic relationships in the text, shedding an additional 6 points of Test PPL.  
* Lowering the gradient clipping threshold (clip \= 0.1) ultimately stabilized the final parameter updates alongside AdamW, arriving at the absolute best performance.

**Target Achievement:** The final optimized configuration (LSTM\_Dropout\_AdamW\_Final) achieved a **Test Perplexity of 115.732**. This dramatically exceeds the mandatory project requirement of achieving, firmly demonstrating a highly performant and stable language model.
