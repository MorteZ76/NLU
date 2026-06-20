import os
import math
import copy
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from typing import Callable, Dict, Any, List, Tuple
from tqdm import tqdm

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


def grid_search(param_grid: Dict[str, List[Any]],
                base_config: Dict[str, Any],
                run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                results_dir: str = "bin/grid_search") -> List[Dict[str, Any]]:
    """
    Perform grid search over specified hyperparameter values with enhanced tracking and visualization.

    Args:
        param_grid: dict mapping hyperparameter names to lists of values to try.
        base_config: base configuration dict that will be copied and updated per trial.
        run_fn: callable that accepts a config dict, runs a single experiment, and
                returns a tuple (metric, extras) where metric is a scalar to minimize
                (e.g. validation PPL) and extras is an informational dict.
        results_dir: directory where a summary JSON of all trials will be saved.

    Returns:
        A list of result dicts sorted by the metric (ascending).
    """
    os.makedirs(results_dir, exist_ok=True)

    # Build list of (param, values) and produce cartesian product
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    total_trials = len(combos)

    print(f"\n{'='*70}")
    print(f"[Grid Search] Starting hyperparameter grid search")
    print(f"[Grid Search] Total combinations: {total_trials}")
    print(f"[Grid Search] Parameters being tuned: {', '.join(keys)}")
    print(f"[Grid Search] Results directory: {results_dir}")
    print(f"{'='*70}\n")

    results: List[Dict[str, Any]] = []
    failed_trials = []
    
    with tqdm(total=total_trials, desc="Grid Search Progress", unit="trial") as pbar:
        for idx, combo in enumerate(combos, start=1):
            trial_cfg = copy.deepcopy(base_config)
            trial_name_parts = [trial_cfg.get("experiment_name", "exp")]
            
            # Build trial config with current combo values
            param_str_parts = []
            for k, v in zip(keys, combo):
                trial_cfg[k] = v
                trial_name_parts.append(f"{k}={v}")
                param_str_parts.append(f"{k}={v}")
            
            trial_cfg["experiment_name"] = "grid_" + "__".join(trial_name_parts)
            param_str = ", ".join(param_str_parts)

            # Update progress bar with current trial info
            pbar.set_description(f"Trial {idx}/{total_trials} | {param_str[:40]}")
            
            try:
                metric, extras = run_fn(trial_cfg)
                status = "✓ PASS" if metric != float('inf') else "✗ FAIL"
            except Exception as e:
                print(f"\n[Grid] Trial {idx} failed with error: {str(e)[:100]}")
                metric = float('inf')
                extras = {"error": str(e)}
                status = "✗ ERROR"
                failed_trials.append({"trial": idx, "error": str(e)})

            result = {
                "trial": idx,
                "config": trial_cfg,
                "metric": metric,
                "extras": extras,
                "status": status,
            }
            results.append(result)

            # Save intermediate results periodically
            with open(os.path.join(results_dir, "latest_results.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            
            pbar.update(1)

    # Sort ascending by metric (lower is better)
    results_sorted = sorted(results, key=lambda r: r["metric"]) 

    # Persist final results
    with open(os.path.join(results_dir, "grid_search_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, indent=2, default=str)

    # Generate results summary
    successful_results = [r for r in results_sorted if r["metric"] != float('inf')]
    
    print(f"\n{'='*70}")
    print(f"[Grid Search] Complete!")
    print(f"[Grid Search] Total trials: {total_trials} | Successful: {len(successful_results)} | Failed: {len(failed_trials)}")
    
    if successful_results:
        best_result = successful_results[0]
        worst_result = successful_results[-1]
        avg_metric = np.mean([r["metric"] for r in successful_results])
        std_metric = np.std([r["metric"] for r in successful_results])
        
        print(f"\n[Results Summary]")
        print(f"  Best metric:   {best_result['metric']:.4f}")
        print(f"  Worst metric:  {worst_result['metric']:.4f}")
        print(f"  Mean metric:   {avg_metric:.4f} (±{std_metric:.4f})")
        print(f"\n[Top 3 Configurations]")
        for rank, result in enumerate(successful_results[:3], 1):
            config = result["config"]
            print(f"  {rank}. Metric: {result['metric']:.4f} | {config.get('experiment_name', 'unnamed')}")
            for k in keys:
                print(f"     - {k}: {config.get(k)}")
    
    if failed_trials:
        print(f"\n[Failed Trials] ({len(failed_trials)} trials)")
        for failed in failed_trials[:3]:
            print(f"  Trial {failed['trial']}: {failed['error'][:60]}")
    
    print(f"[Grid Search] Results saved to: {results_dir}")
    print(f"{'='*70}\n")
    
    # Attempt to generate visualization
    try:
        _plot_grid_search_results(results_sorted, results_dir, keys)
    except Exception as e:
        print(f"[Grid Search] Could not generate visualization: {e}")
    
    return results_sorted


def _plot_grid_search_results(results: List[Dict[str, Any]], results_dir: str, param_keys: List[str]):
    """
    Generate visualization plots of grid search results.
    
    Args:
        results: sorted list of result dicts
        results_dir: directory to save plots
        param_keys: list of hyperparameter names being tuned
    """
    successful_results = [r for r in results if r["metric"] != float('inf')]
    
    if not successful_results:
        return
    
    try:
        metrics = [r["metric"] for r in successful_results]
        trial_numbers = list(range(1, len(successful_results) + 1))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Metric over trials (sorted)
        axes[0].plot(trial_numbers, metrics, marker='o', linestyle='-', linewidth=2, markersize=4)
        axes[0].set_xlabel("Trial Rank (sorted by metric)")
        axes[0].set_ylabel("Metric Value")
        axes[0].set_title("Grid Search Results: Metric by Trial")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=min(metrics), color='r', linestyle='--', alpha=0.5, label=f"Best: {min(metrics):.4f}")
        axes[0].legend()
        
        # Plot 2: Distribution of metrics
        axes[1].hist(metrics, bins=max(10, len(successful_results)//3), edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Metric Value")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Metrics Across Trials")
        axes[1].axvline(x=np.mean(metrics), color='r', linestyle='--', label=f"Mean: {np.mean(metrics):.4f}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "grid_search_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Grid Search] Visualization saved to: {plot_path}")
    except Exception as e:
        print(f"[Grid Search] Visualization generation failed: {e}")