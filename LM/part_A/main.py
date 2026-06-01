import sys
import os
import json
import math
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader

# Import modular project blocks
from utils import download_ptb, read_file, Lang, PennTreeBank, collate_fn
from model import LM_RNN, LM_LSTM
from functions import set_seed, init_weights, train_loop, eval_loop, save_experiment

def main():
    # 0. Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"[Config] Loaded configuration from {config_path}")
    print(f"[Config] Experiment: {config['experiment_name']} | Model: {config['model_type']} | Optimizer: {config['optimizer']}")

    # 1. Command-Line Argument Interface Setup
    parser = argparse.ArgumentParser(description="Autoregressive Language Model Training & Evaluation.")
    parser.add_argument(
        "--eval_only", 
        action="store_true", 
        help="Skip the training loop to run evaluation directly using a saved checkpoint."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=f"bin/{config['experiment_name']}/{config['experiment_name']}.pt", 
        help="Relative path to a pre-trained PyTorch weight checkpoint (.pt)."
    )
    args = parser.parse_args()

    # 2. Seeding & Hardware Configuration
    set_seed(1234)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Initializing pipeline on execution device: {device}")

    # 3. Dataset Download & Loading
    download_ptb()  # Retrieve Penn TreeBank source files automatically if missing
    
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Construct language vocabulary based exclusively on the training corpus
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    pad_idx = lang.word2id["<pad>"]
    vocab_len = len(lang.word2id)
    print(f"[Dataset] Vocabulary constructed with: {vocab_len} unique tokens.")

    # Initialize PyTorch Map-style Dataset split classes
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Initialize Validation and Test DataLoaders
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config['batch_size'], 
        collate_fn=partial(collate_fn, pad_token=pad_idx, device=device)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        collate_fn=partial(collate_fn, pad_token=pad_idx, device=device)
    )

    # Core Configuration Parameters
    experiment_name = config['experiment_name']
    model_type = config['model_type']
    optimizer_name = config['optimizer']
    emb_size = config['emb_size']
    hid_size = config['hidden_size']
    lr = config['lr']
    patience = config['patience']
    batch_size = config['batch_size']
    clip = config['clip']
    n_epochs = config['n_epochs']

    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

    # ================= EVALUATION ONLY MODE =================
    if args.eval_only:
        if not os.path.exists(args.model_path):
            print(f"\n[Error] Saved model checkpoint not found at: {args.model_path}")
            print("Verify checkpoint path or execute standard training first.")
            sys.exit(1)
            
        print(f"\n=== Evaluation Mode ===")
        print(f"[Load] Loading model parameters from: {args.model_path}")
        
        # Instantiate model architecture and load saved parameters
        if model_type.upper() == 'LSTM':
            model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=pad_idx).to(device)
        else:
            model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=pad_idx).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Calculate perplexity metrics across validation and test sets
        val_ppl, _ = eval_loop(dev_loader, criterion_eval, model)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        
        print(f"\n{'='*48}")
        print(f"Pre-Trained Model Evaluation Metrics:")
        print(f"Validation Perplexity (PPL): {val_ppl:.3f}")
        print(f"Test Set Perplexity (PPL):   {test_ppl:.3f}")
        print(f"{'='*48}")
        return

    # ================= STANDARD TRAINING MODE =================
    print("\n=== Training & Optimization Mode ===")
    
    # Training structures are loaded only if the pipeline is in training mode
    train_dataset = PennTreeBank(train_raw, lang)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(collate_fn, pad_token=pad_idx, device=device), 
        shuffle=True
    )

    # Build model architecture and apply custom weight initializations
    if model_type.upper() == 'LSTM':
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=pad_idx).to(device)
    else:
        model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=pad_idx).to(device)
    model.apply(init_weights)

    if optimizer_name.upper() == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.upper() == 'ADAMW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Training Loop Tracking Variables
    losses_train = []
    losses_dev = []
    
    best_ppl = math.inf
    best_model_state = None
    current_patience = patience  # Initialize dynamic early stopping patience tracking steps

    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch Progress")
    for epoch in pbar:
        # Run training epoch
        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        losses_train.append(train_loss)
        
        # Run validation epoch
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        losses_dev.append(loss_dev)
        
        pbar.set_description(f"Epoch {epoch} | Val PPL: {ppl_dev:.2f}")

        # Update best model weights or decrement early stopping patience
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            current_patience = patience  # Reset validation tracking steps
        else:
            current_patience -= 1
            if current_patience <= 0:
                print(f"\n[Early Stopping] Triggered at Epoch {epoch} due to plateauing validation loss.")
                break

    # Restore the best performing parameters from the training run
    if model_type.upper() == 'LSTM':
        best_model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=pad_idx).to(device)
    else:
        best_model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=pad_idx).to(device)
    best_model.load_state_dict(best_model_state)

    # Final evaluate run across testing dataset
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print(f"\n{'='*48}\nTest Set Evaluation Results:\n{model_type} Perplexity (PPL): {final_ppl:.3f}\n{'='*48}")

    # Structuring and Saving Run Metadata & Metrics
    config_details = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "optimizer": optimizer_name,
        "emb_size": emb_size,
        "hidden_size": hid_size,
        "lr": lr,
        "patience": patience,
        "batch_size": batch_size,
        "clip": clip,
        "n_epochs": n_epochs,
        "vocabulary_size": vocab_len,
        "best_val_ppl": best_ppl,
        "final_test_ppl": final_ppl
    }
    
    save_experiment(
        model=best_model,
        hyperparameters=config_details,
        train_losses=losses_train,
        dev_losses=losses_dev,
        name=experiment_name
    )

if __name__ == "__main__":
    main()