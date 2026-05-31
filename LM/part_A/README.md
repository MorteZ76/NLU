Language Modeling (LM) - Part 1.A

This repository contains the clean, fully-modularized implementation of Part 1.A (Baseline Vanilla Recurrent Neural Network Language Model) evaluated on the Penn TreeBank (PTB) dataset.

The model is trained autoregressively to predict the next word token by optimizing cross-entropy loss over shifted inputs.

File and Directory Structure

LM/
└── Part_A/
    ├── utils.py          # Data loaders, vocabulary generation, and dynamic batch collators
    ├── model.py          # PyTorch model definition (LM_RNN)
    ├── functions.py      # Seeding, parameter initializers, training/evaluation loops, and save utilities
    ├── main.py           # Argument-driven program execution orchestrator
    ├── README.md         # Academic documentation and execution guides
    ├── dataset/          # Automated storage directory for the downloaded PTB corpus
    └── bin/              # Output directory for serialized checkpoints and analytics
        └── baseline_rnn/
            ├── baseline_rnn.pt    # PyTorch state-dict parameters
            ├── config.json        # Recorded hyperparameters and performance summary
            └── loss_plot.png      # Training vs. Validation Loss curve diagram


Features & Project Structure

Academic Commenting Guidelines:
Code components, variable structures, tensor shapes, mathematical definitions, and loop transitions are documented with clean, professional, and descriptive comments.

Guaranteed Reproducibility:
Fixed seed propagation of 1234 ensures identical parameter initialization, data batch shuffling, and calculation outputs.

Advanced Parameter Initializations:
Uses custom initializations (Xavier uniform for input projections, orthogonal initialization for RNN hidden recurrent connections to mitigate standard vanishing and exploding gradients, and uniform initialization for linear projection decoders).

Execution Mode Flexibility:
Supports a train-and-save execution or loading pre-trained checkpoints for fast testing/evaluation without needing to re-train the model.

Technical Metrics & Mathematics

Autoregressive Dataset Shifts

Given a natural sequence of tokens $w_1, w_2, \dots, w_N$, the mapping defines:

Inputs (X): $[w_1, w_2, \dots, w_{N-1}]$

Targets (Y): $[w_2, w_3, \dots, w_N]$

Perplexity (PPL)

Evaluation metrics calculate the exponential of average Cross-Entropy loss over all non-padded tokens to determine Perplexity:


$$PPL = \exp\left(\frac{1}{M}\sum_{i=1}^{M}\mathcal{L}_i\right) = e^{\text{Average Cross-Entropy Loss}}$$

Execution Guidelines

First, navigate into the subdirectory containing Part A files:

cd LM/Part_A


Action Mode 1: Run Full Training and Optimization

Runs the full training dataset with validation tracking and saves the model binary, configurations, and loss curves inside the /bin directory:

python main.py


Action Mode 2: Run Evaluation-Only

Loads a saved model checkpoint and outputs its Validation and Test perplexities without starting a training run:

python main.py --eval_only --model_path bin/baseline_rnn/baseline_rnn.pt


Command-line Arguments Reference:

--eval_only: Skip standard training and run direct evaluations.

--model_path: Custom location pointing to the targeted checkpoint. (Default: bin/baseline_rnn/baseline_rnn.pt)

Dependencies

Python 3.8+

PyTorch 1.10+

NumPy

Matplotlib

tqdm