import os
import math
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def set_seed(seed=1234):
    """
    Locks system random number generators across Python, NumPy, and PyTorch (CPU + GPU backend)
    to guarantee identical parameter initializations and data batching behaviors.
    
    Args:
        seed (int): The targeted environment seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic algorithms inside CUDA execution environments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Init] Reproducibility seed successfully configured to: {seed}")

def init_weights(mat):
    """
    Applies professional, custom parameter initializations.
    
    Recurrent Matrices (RNN, LSTM, GRU):
      - Input-to-hidden matrices ('weight_ih'): Xavier Uniform
      - Hidden-to-hidden matrices ('weight_hh'): Orthogonal initialization (stabilizes gradients)
      - Biases ('bias'): Set to zero
      
    Linear Layers:
      - Weights initialized uniformly between [-0.01, 0.01]
      - Biases initialized to 0.01
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        if mul > 0:
                            torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                        else:
                            torch.nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        if mul > 0:
                            torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                        else:
                            torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

def train_loop(data_loader, optimizer, criterion, model, clip=5):
    """
    Runs a single training epoch across the input DataLoader.
    
    Tracks losses dynamically against non-padded elements to compute 
    an accurate representation of global dataset cross-entropy loss.
    
    Args:
        data_loader (DataLoader): Training split batched dataset loader.
        optimizer (Optimizer): The selected optimization method (e.g. SGD).
        criterion (Module): Training loss metric (CrossEntropyLoss ignoring PAD).
        model (Module): Language modeling neural network.
        clip (float): Maximum norm constraint for gradient clipping.
        
    Returns:
        float: Average Cross-Entropy loss calculated over all parsed tokens.
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data_loader:
        optimizer.zero_grad()
        
        # Forward Pass
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        
        # Track raw loss scaled by batch tokens
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        
        # Backward Pass
        loss.backward()
        
        # Mitigate exploding gradients via norm clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array) / sum(number_of_tokens)

def eval_loop(data_loader, eval_criterion, model):
    """
    Evaluates the model across validation or testing partitions.
    
    Computes Perplexity (PPL) using the mathematical formula:
        PPL = exp(Total Cross-Entropy Loss / Total Valid Tokens)
        
    Args:
        data_loader (DataLoader): The evaluation split dataset loader.
        eval_criterion (Module): Evaluation loss metric (reduction='sum' to aggregate exact values).
        model (Module): Loaded language modeling neural network.
        
    Returns:
        tuple (float, float): Measured Perplexity (PPL) and token-averaged cross-entropy loss.
    """
    model.eval()
    loss_array = []
    number_of_tokens = []
    
    with torch.no_grad():
        for sample in data_loader:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            
            # Record absolute accumulated losses and active tokens
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    total_tokens = sum(number_of_tokens)
    total_loss = sum(loss_array)
    
    # Calculate perplexity using natural exponentiation of average loss
    ppl = math.exp(total_loss / total_tokens)
    average_loss = total_loss / total_tokens
    return ppl, average_loss

def save_experiment(model, hyperparameters, train_losses, dev_losses, name="baseline_rnn"):
    """
    Persists training progress by saving:
      - Model parameters (.pt state dict)
      - Hyperparameters configuration (config.json)
      - Train/Validation Loss curves (loss_plot.png)
      
    Args:
        model (Module): Best performing state dictionary model.
        hyperparameters (dict): Relevant experiment variables.
        train_losses (list of float): Recorded training epoch losses.
        dev_losses (list of float): Recorded validation epoch losses.
        name (str): Identifier name for the target experiment save folder.
    """
    os.makedirs("bin", exist_ok=True)
    exp_dir = os.path.join("bin", name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 1. Save model parameters
    model_path = os.path.join(exp_dir, f"{name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[Save] Model state dict saved to: {model_path}")
    
    # 2. Export configuration and results log
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"[Save] Run configuration saved to: {config_path}")
    
    # 3. Save loss curves plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="#1f77b4", marker='o', linewidth=2)
    plt.plot(dev_losses, label="Val Loss", color="#ff7f0e", marker='s', linestyle="--", linewidth=2)
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.title(f"Loss Curves - {name}", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, shadow=False)
    
    plot_path = os.path.join(exp_dir, "loss_plot.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Save] Loss plot visualization saved to: {plot_path}")