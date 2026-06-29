import os
import math
import copy
import json
import random
from datetime import datetime
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


def grid_search(param_name: str,
                param_values: List[Any],
                base_config: Dict[str, Any],
                run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                results_dir: str = "bin/grid_search") -> Dict[str, Any]:
    """
    Perform grid search over a single hyperparameter with organized result storage.
    
    Args:
        param_name: name of the hyperparameter being tuned
        param_values: list of values to try for this parameter
        base_config: base configuration dict that will be copied and updated per trial
        run_fn: callable that accepts a config dict, runs experiment, returns (metric, extras)
        results_dir: directory to save grid search results

    Returns:
        Dictionary with best config and results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    total_trials = len(param_values)
    
    print(f"\n{'='*70}")
    print(f"[Grid Search] Tuning parameter: {param_name}")
    print(f"[Grid Search] Values to test: {param_values}")
    print(f"[Grid Search] Total trials: {total_trials}")
    print(f"[Grid Search] Results directory: {results_dir}")
    print(f"{'='*70}\n")

    results: List[Dict[str, Any]] = []
    failed_trials = []
    
    with tqdm(total=total_trials, desc=f"Searching {param_name}", unit="trial") as pbar:
        for idx, param_val in enumerate(param_values, start=1):
            trial_cfg = copy.deepcopy(base_config)
            trial_cfg[param_name] = param_val
            trial_cfg["experiment_name"] = f"grid_{param_name}={param_val}"
            
            pbar.set_description(f"Trial {idx}/{total_trials} | {param_name}={param_val}")
            
            try:
                metric, extras = run_fn(trial_cfg)
                status = "✓ PASS" if metric != float('inf') else "✗ FAIL"
            except Exception as e:
                print(f"\n[Grid] Trial {idx} failed: {str(e)[:80]}")
                metric = float('inf')
                extras = {"error": str(e)}
                status = "✗ ERROR"
                failed_trials.append({"trial": idx, "value": param_val, "error": str(e)})

            result = {
                "trial": idx,
                "param_value": param_val,
                "metric": metric,
                "extras": extras,
                "status": status,
            }
            results.append(result)
            pbar.update(1)

    # Sort by metric (ascending - lower is better)
    results_sorted = sorted(results, key=lambda r: r["metric"])
    successful_results = [r for r in results_sorted if r["metric"] != float('inf')]
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"[Grid Search] Complete for parameter: {param_name}")
    print(f"[Grid Search] Successful: {len(successful_results)} | Failed: {len(failed_trials)}")
    
    if successful_results:
        best = successful_results[0]
        metrics = [r["metric"] for r in successful_results]
        print(f"\n[Results Summary]")
        print(f"  Best metric:   {best['metric']:.4f} (at {param_name}={best['param_value']})")
        print(f"  Mean metric:   {np.mean(metrics):.4f}")
        print(f"  Std metric:    {np.std(metrics):.4f}")
    
    print(f"{'='*70}\n")
    
    # Generate report and plot
    try:
        _generate_single_param_report(results_sorted, results_dir, param_name, successful_results)
    except Exception as e:
        print(f"[Grid Search] Could not generate report: {e}")
    
    try:
        _plot_parameter_performance(results_sorted, results_dir, param_name)
    except Exception as e:
        print(f"[Grid Search] Could not generate plot: {e}")
    
    return {
        "best_config": {param_name: successful_results[0]["param_value"]} if successful_results else {},
        "best_metric": successful_results[0]["metric"] if successful_results else float('inf'),
        "results": results_sorted
    }


def _generate_single_param_report(results: List[Dict[str, Any]], 
                                  results_dir: str, 
                                  param_name: str,
                                  successful_results: List[Dict[str, Any]]):
    """Generate text report for single parameter grid search."""
    report_path = os.path.join(results_dir, "grid_search_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"GRID SEARCH REPORT: {param_name.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameter: {param_name}\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Trials:       {len(results)}\n")
        f.write(f"Successful Trials:  {len(successful_results)}\n")
        f.write(f"Failed Trials:      {len(results) - len(successful_results)}\n\n")
        
        if successful_results:
            metrics = [r["metric"] for r in successful_results]
            f.write(f"Best Metric:        {min(metrics):.6f}\n")
            f.write(f"Worst Metric:       {max(metrics):.6f}\n")
            f.write(f"Mean Metric:        {np.mean(metrics):.6f}\n")
            f.write(f"Std Dev Metric:     {np.std(metrics):.6f}\n")
            f.write(f"Median Metric:      {np.median(metrics):.6f}\n\n")
        
        # All results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n")
        for rank, result in enumerate(successful_results, 1):
            f.write(f"\n#{rank} | {param_name}={result['param_value']} | Metric: {result['metric']:.6f} | Status: {result['status']}\n")
            if result.get('extras'):
                for key, val in result['extras'].items():
                    if key != 'error':
                        f.write(f"     {key}: {val}\n")
    
    print(f"[Grid Search] Report saved to: {report_path}")


def _plot_parameter_performance(results: List[Dict[str, Any]], 
                               results_dir: str, 
                               param_name: str):
    """Plot parameter values vs performance metric."""
    successful_results = [r for r in results if r["metric"] != float('inf')]
    
    if not successful_results:
        return
    
    try:
        param_vals = [r["param_value"] for r in successful_results]
        metrics = [r["metric"] for r in successful_results]
        
        # Convert to string for plotting (handles various types)
        param_labels = [str(v) for v in param_vals]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot with markers
        ax.plot(range(len(param_labels)), metrics, marker='o', linestyle='-', 
               linewidth=2.5, markersize=8, color='#1f77b4', label='Performance')
        
        # Highlight best result
        best_idx = np.argmin(metrics)
        ax.scatter([best_idx], [metrics[best_idx]], color='red', s=300, zorder=5, 
                  marker='*', label=f'Best: {metrics[best_idx]:.4f}')
        
        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=45, ha='right')
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel("Metric Value", fontsize=12, fontweight='bold')
        ax.set_title(f"Grid Search: {param_name} vs Performance", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "parameter_performance.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Grid Search] Plot saved to: {plot_path}")
    except Exception as e:
        print(f"[Grid Search] Plot generation failed: {e}")


def sequential_grid_search(param_tuning_order: List[Dict[str, Any]],
                          base_config: Dict[str, Any],
                          run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                          base_results_dir: str = "bin/grid_search") -> Dict[str, Any]:
    """
    Perform sequential grid search, tuning one parameter at a time.
    
    Fixes the best value of each parameter before moving to the next.
    
    Args:
        param_tuning_order: List of dicts, each with keys:
            - "name": parameter name
            - "values": list of values to try
            Example: [
                {"name": "batch_size", "values": [32, 64, 128]},
                {"name": "hidden_size", "values": [150, 200, 300]},
                {"name": "lr", "values": [0.001, 0.01, 0.1]}
            ]
        base_config: base configuration dict
        run_fn: callable for running single experiment
        base_results_dir: base directory for all grid search results
    
    Returns:
        Dictionary with final best config and all search results
    """
    # Create timestamped parent directory for all grid searches
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_results_dir = os.path.join(base_results_dir, f"sequential__{timestamp}")
    os.makedirs(parent_results_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"[Sequential Grid Search] Starting sequential hyperparameter tuning")
    print(f"[Sequential Grid Search] Parameters to tune in order:")
    for i, param_cfg in enumerate(param_tuning_order, 1):
        print(f"  {i}. {param_cfg['name']}: {param_cfg['values']}")
    print(f"{'='*70}\n")
    
    # Start with base config
    current_best_config = copy.deepcopy(base_config)
    all_search_results = {}
    
    # Tune each parameter sequentially
    for step, param_cfg in enumerate(param_tuning_order, 1):
        param_name = param_cfg["name"]
        param_values = param_cfg["values"]
        
        print(f"\n[Sequential] Step {step}/{len(param_tuning_order)}: Tuning {param_name}")
        print(f"[Sequential] Current best config: {current_best_config}")
        
        # Create subdirectory for this parameter's search
        param_results_dir = os.path.join(parent_results_dir, f"{step:02d}_{param_name}")
        os.makedirs(param_results_dir, exist_ok=True)
        
        # Run grid search for this parameter with fixed other parameters
        search_result = grid_search(
            param_name=param_name,
            param_values=param_values,
            base_config=current_best_config,
            run_fn=run_fn,
            results_dir=param_results_dir
        )
        
        all_search_results[param_name] = search_result
        
        # Update current best config with the best value found for this parameter
        if search_result["best_config"]:
            current_best_config.update(search_result["best_config"])
            print(f"[Sequential] Fixed {param_name}={search_result['best_config'][param_name]} " 
                  f"(metric: {search_result['best_metric']:.4f})")
    
    # Generate final summary report
    _generate_sequential_summary(parent_results_dir, all_search_results, current_best_config)
    
    print(f"\n{'='*70}")
    print(f"[Sequential Grid Search] Complete!")
    print(f"[Sequential Grid Search] Final best config: {current_best_config}")
    print(f"[Sequential Grid Search] Results saved to: {parent_results_dir}")
    print(f"{'='*70}\n")
    
    return {
        "best_config": current_best_config,
        "all_results": all_search_results,
        "results_dir": parent_results_dir
    }


def _generate_sequential_summary(parent_dir: str, 
                                 all_results: Dict[str, Any],
                                 final_config: Dict[str, Any]):
    """Generate summary report of sequential grid search."""
    summary_path = os.path.join(parent_dir, "sequential_summary.txt")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SEQUENTIAL GRID SEARCH SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("FINAL BEST CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, val in final_config.items():
            f.write(f"  {key}: {val}\n")
        
        f.write("\n\nSEARCH PROGRESSION\n")
        f.write("-" * 80 + "\n")
        for step, (param_name, result) in enumerate(all_results.items(), 1):
            f.write(f"\nStep {step}: {param_name}\n")
            f.write(f"  Best Value: {result['best_config'].get(param_name, 'N/A')}\n")
            f.write(f"  Best Metric: {result['best_metric']:.6f}\n")
            f.write(f"  Total Trials: {len(result['results'])}\n")
    
    print(f"[Sequential] Summary saved to: {summary_path}")
    
def check_nt_asgd_trigger(val_losses, non_mono=3):
    """
    Checks the non-monotonic trigger condition for switching from SGD to ASGD.

    Trigger condition: current validation loss is worse than the minimum loss
    recorded over all epochs except the last `non_mono` epochs. This detects
    a plateau and signals that averaging (ASGD) will be more beneficial than
    continued SGD updates.

    Args:
        val_losses (list of float): Validation losses recorded per epoch so far.
        non_mono (int): Number of recent epochs excluded from the minimum comparison.

    Returns:
        bool: True if the NT-ASGD trigger condition is met, False otherwise.
    """
    if len(val_losses) > non_mono:
        if val_losses[-1] > min(val_losses[:-non_mono]):
            return True
    return False

def swap_asgd_weights(model, optimizer):
    """
    Temporarily replaces model parameters with ASGD's accumulated averaged weights ('ax').

    Used during validation when ASGD is active, so that evaluation reflects the
    averaged parameters rather than the current noisy SGD iterate.

    Args:
        model (nn.Module): The model whose parameters will be temporarily swapped.
        optimizer (optim.ASGD): The active ASGD optimizer holding averaged weights.

    Returns:
        dict: Mapping from each parameter tensor to its original (pre-swap) data,
              used to restore weights after evaluation via restore_asgd_weights().
    """
    backup = {}
    for prm in model.parameters():
        if prm in optimizer.state and 'ax' in optimizer.state[prm]:
            backup[prm] = prm.data.clone()
            prm.data = optimizer.state[prm]['ax'].clone()
    return backup

def restore_asgd_weights(model, backup):
    """
    Restores model parameters to their pre-swap values after ASGD evaluation.

    Args:
        model (nn.Module): The model whose parameters will be restored.
        backup (dict): Parameter-to-data mapping returned by swap_asgd_weights().
    """
    for prm, backup_weight in backup.items():
        prm.data = backup_weight.clone()