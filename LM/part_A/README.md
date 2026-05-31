Language Modeling (LM) - Part 1.A

This repository contains the clean, fully-modularized implementation of Part 1.A (Baseline Vanilla Recurrent Neural Network Language Model) evaluated on the Penn TreeBank (PTB) dataset.

The baseline model is trained autoregressively to predict the next word token by optimizing cross-entropy loss over shifted inputs.

Assignment Requirements & Guidelines

As instructed by the Teaching Assistants (TAs), this sub-folder is designed to fulfill the following strict operational and academic criteria:

Modular Architecture:
The code must be completely separated into distinct files (utils.py, model.py, functions.py, and main.py) inside this subdirectory.

Mandatory Performance Threshold:
Through hyperparameter optimization and incremental modifications, the final achieved test perplexity must be strictly below 250 ($PPL < 250$).

Incremental Experimentation:
Modifications to the baseline architecture must be added one at a time. If a specific technique degrades performance, it can be removed before testing the next modification. However, both successful and unsuccessful experiments must be recorded and commented on in the final report.

Hyperparameter Tuning:
Active optimization of critical parameters (specifically the learning rate, embedding/hidden dimensions, and batch sizes) is required to minimize Perplexity.

No Notebooks:
Only clean, bug-free, and well-documented Python scripts are accepted.

File and Directory Structure

LM/
└── Part_A/
    ├── utils.py          # Data preprocessors, Lang vocabulary class, and batch collators
    ├── model.py          # PyTorch model definition (LM_RNN)
    ├── functions.py      # Seeding, weight initializers, train/validation loops, and plot utilities
    ├── main.py           # Command-line-driven execution orchestrator
    ├── requirements.txt  # Project library dependencies
    ├── README.md         # This academic documentation, execution guide, and experimental log
    ├── dataset/          # Stored raw Penn TreeBank text split files
    └── bin/              # Saved model checkpoints, config parameters, and metrics
        └── baseline_rnn/
            ├── baseline_rnn.pt    # PyTorch model state-dict checkpoint
            ├── config.json        # Recorded hyperparameters and final metrics
            └── loss_plot.png      # Training vs. Validation Loss curves


Technical Features

Strict Seeding for Reproducibility:
A fixed global seed of 1234 is applied across all random number generators (Python, NumPy, and PyTorch CPU/CUDA backends) to ensure deterministic weights initialization and sequence batching.

Smart Parameter Initializations:
Applies professional initializations (Xavier Uniform for input-to-hidden projections, Orthogonal Initialization for hidden-to-hidden recurrent weights to mitigate gradient instability, and Uniform bounding for linear decoding layers).

Flexible Action Modes:
Supports running a full training loop with early stopping or loading pre-trained parameters directly to evaluate performance on the test set.

Mathematical Foundations

Autoregressive Dataset Shifts

For a natural sequence of tokens $w_1, w_2, \dots, w_N$, the training configuration aligns:

Inputs (X): $[w_1, w_2, \dots, w_{N-1}]$

Targets (Y): $[w_2, w_3, \dots, w_N]$

Perplexity (PPL)

The core metric evaluated is Perplexity, which mathematically represents the exponential of the average Cross-Entropy loss computed over all non-padded tokens ($M$ total tokens):

$$PPL = \exp\left(\frac{1}{M}\sum_{i=1}^{M}\mathcal{L}_i\right) = e^{\text{Average Cross-Entropy Loss}}$$

Installation & Setup

Before running the model, make sure to install all the necessary dependencies. Navigate to the Part_A directory and install the packages using the provided requirements.txt file:

cd LM/Part_A
pip install -r requirements.txt


This will automatically install compatible versions of torch, numpy, matplotlib, and tqdm.

Execution Guidelines

Ensure you are in the correct subdirectory before executing:

cd LM/Part_A


Option 1: Standard Training & Optimization

Runs the full training corpus, validates performance, and outputs the baseline artifacts to the /bin directory:

python main.py


Option 2: Direct Model Evaluation

Skips training to immediately evaluate a saved model binary on the validation and test datasets:

python main.py --eval_only --model_path bin/baseline_rnn/baseline_rnn.pt


Experimental Log & Results (Part 1.A Report)

Below is the record of incremental architectural modifications, hyperparameter configurations, and their corresponding validation and test Perplexities.

Exp ID

Model Architecture Description

Learning Rate ($lr$)

Batch Size

Validation PPL

Test PPL

Status / Action

0

Vanilla Baseline RNN (LM_RNN)

1.0

64



$$Insert Val$$





$$Insert Test$$



Baseline

1

Baseline + 

$$Technique A$$



1.0

64



$$Insert$$





$$Insert$$





$$Kept / Rejected$$



2

Baseline + 

$$Technique A$$

 + 

$$Technique B$$



0.5

64



$$Insert$$





$$Insert$$





$$Kept / Rejected$$



Experimental Observations & Discussion

Hyperparameter Optimization: 

$$Discuss how changing the learning rate, optimizer, batch size, or hidden layers affected convergence stability and final PPL values.$$

Unsuccessful Modifications: 

$$Document any attempted modification that degraded model performance, discussing why it may have led to gradient issues, overfitting, or underfitting on the PTB corpus.$$

Target Achievement: 

$$Confirm here that the final model meets the mandatory project requirement of achieving* $PPL < 250$*.$$