import os
import copy
import json
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, List, Tuple
from tqdm import tqdm

try:
    from conll import evaluate
except ImportError:
    print("[Warning] 'conll' module not found. F1 Evaluation may fail if not provided.")
    
from sklearn.metrics import classification_report

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Init] Reproducibility seed successfully configured to: {seed}")

def init_weights(mat):
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

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses.
        
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
        
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterances'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
                
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def save_experiment(model, hyperparameters, train_losses, dev_metrics, name="baseline_atis"):
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
        
    # 3. Visualization (Loss vs Epochs)
    if train_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", color="#1f77b4", marker='o', linewidth=2)
        if dev_metrics:
            plt.plot(dev_metrics, label="Dev Loss", color="#ff7f0e", marker='s', linestyle="--", linewidth=2)
        plt.xlabel("Epochs", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title(f"Training Loss - {name}", fontsize=13, fontweight='bold')
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.savefig(os.path.join(exp_dir, "loss_plot.png"), dpi=200, bbox_inches='tight')
        plt.close()

def grid_search(param_name: str,
                param_values: List[Any],
                base_config: Dict[str, Any],
                run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                results_dir: str = "bin/grid_search") -> Dict[str, Any]:
    """
    Modified to *maximize* the F1 score instead of minimizing loss/PPL.
    """
    os.makedirs(results_dir, exist_ok=True)
    total_trials = len(param_values)
    results = []
    
    print(f"\n{'='*70}")
    print(f"[Grid Search] Tuning parameter: {param_name} | Values: {param_values}")
    
    with tqdm(total=total_trials, desc=f"Searching {param_name}", unit="trial") as pbar:
        for idx, param_val in enumerate(param_values, start=1):
            trial_cfg = copy.deepcopy(base_config)
            trial_cfg[param_name] = param_val
            trial_cfg["experiment_name"] = f"grid_{param_name}={param_val}"
            
            try:
                # Expecting F1 score as primary metric (higher is better)
                f1_metric, extras = run_fn(trial_cfg)
                status = "✓ PASS" if f1_metric != -1.0 else "✗ FAIL"
            except Exception as e:
                print(f"\n[Grid] Trial {idx} failed: {str(e)[:80]}")
                f1_metric = -1.0
                extras = {"error": str(e)}
                status = "✗ ERROR"

            results.append({
                "trial": idx, "param_value": param_val,
                "metric": f1_metric, "extras": extras, "status": status,
            })
            pbar.update(1)

    # Sort DESCENDING (Higher F1 is better)
    results_sorted = sorted(results, key=lambda r: r["metric"], reverse=True)
    successful_results = [r for r in results_sorted if r["metric"] != -1.0]
    
    if successful_results:
        best = successful_results[0]
        print(f"\n[Results Summary] Best metric: {best['metric']:.4f} (at {param_name}={best['param_value']})")
    print(f"{'='*70}\n")
    
    return {
        "best_config": {param_name: successful_results[0]["param_value"]} if successful_results else {},
        "best_metric": successful_results[0]["metric"] if successful_results else -1.0,
        "results": results_sorted
    }

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