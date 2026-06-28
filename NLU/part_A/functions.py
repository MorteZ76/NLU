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

from sklearn.metrics import classification_report

# Will be downloaded automatically by utils.py
try:
    from conll import evaluate
except ImportError:
    pass

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
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
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
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                # Safe fallback supporting both "utterance" and "utterances" key mappings
                utt_key = 'utterances' if 'utterances' in sample else 'utterance'
                utt_ids = sample[utt_key][id_seq][:length].tolist()
                # utt_ids = sample['utterances'][id_seq][:length].tolist()
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
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def save_experiment(model, hyperparameters, train_losses, dev_metrics, name="baseline_atis",
                    final_scores: Dict[str, Any] = None):
    """
    Saves model checkpoint, config, loss plot, and a human-readable results_summary.txt.

    Args:
        final_scores: dict with keys such as 'best_dev_f1', 'test_slot_f1', 'test_intent_acc',
                      'stopped_at_epoch'. All optional — only present keys are written.
    """
    os.makedirs("bin", exist_ok=True)
    exp_dir = os.path.join("bin", name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save model parameters
    model_path = os.path.join(exp_dir, f"{name}.pt")
    torch.save(model.state_dict(), model_path)
    
    # Export configuration
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, indent=4)
        
    # Visualization (Loss vs Epochs)
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

    # Human-readable results summary
    summary_path = os.path.join(exp_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EXPERIMENT RESULTS — {name}\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("HYPERPARAMETERS\n" + "-" * 60 + "\n")
        _DISPLAY_KEYS = ["model_type", "optimizer", "emb_size", "hidden_size",
                         "lr", "clip", "n_epochs", "patience", "batch_size"]
        for key in _DISPLAY_KEYS:
            if key in hyperparameters:
                f.write(f"  {key:<20}: {hyperparameters[key]}\n")
        f.write("\n")

        if final_scores:
            f.write("FINAL SCORES\n" + "-" * 60 + "\n")
            score_labels = {
                "best_dev_f1":       "Best Dev Slot F1",
                "test_slot_f1":      "Test Slot F1",
                "test_intent_acc":   "Test Intent Accuracy",
                "dev_intent_acc":    "Dev Intent Accuracy",
                "stopped_at_epoch":  "Stopped at Epoch",
            }
            for key, label in score_labels.items():
                if key in final_scores:
                    val = final_scores[key]
                    fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
                    f.write(f"  {label:<26}: {fmt}\n")
            f.write("\n")

        f.write("FILES\n" + "-" * 60 + "\n")
        f.write(f"  Checkpoint : {model_path}\n")
        f.write(f"  Config     : {config_path}\n")
        if train_losses:
            f.write(f"  Loss plot  : {os.path.join(exp_dir, 'loss_plot.png')}\n")
        f.write(f"  This file  : {summary_path}\n")

    print(f"[Save] Results summary written to: {summary_path}")


def grid_search(param_name: str,
                param_values: List[Any],
                base_config: Dict[str, Any],
                run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                results_dir: str = "bin/grid_search") -> Dict[str, Any]:
    """
    Modified to MAXIMIZE the metric (F1 score).
    """
    os.makedirs(results_dir, exist_ok=True)
    total_trials = len(param_values)
    results = []
    
    with tqdm(total=total_trials, desc=f"Searching {param_name}", unit="trial") as pbar:
        for idx, param_val in enumerate(param_values, start=1):
            trial_cfg = copy.deepcopy(base_config)
            trial_cfg[param_name] = param_val
            trial_cfg["experiment_name"] = f"grid_{param_name}={param_val}"
            
            try:
                # metric is F1
                metric, extras = run_fn(trial_cfg)
                status = "✓ PASS" if metric != -1.0 else "✗ FAIL"
            except Exception as e:
                print(f"\n[Grid] Trial {idx} failed: {str(e)[:80]}")
                metric = -1.0
                extras = {"error": str(e)}
                status = "✗ ERROR"

            results.append({
                "trial": idx, "param_value": param_val,
                "metric": metric, "extras": extras, "status": status,
            })
            pbar.update(1)

    # Sort DESCENDING because higher F1 is better
    results_sorted = sorted(results, key=lambda r: r["metric"], reverse=True)
    successful_results = [r for r in results_sorted if r["metric"] != -1.0]

    # Save per-trial breakdown for this parameter
    trial_log_path = os.path.join(results_dir, "trial_results.txt")
    with open(trial_log_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"GRID SEARCH — Parameter: {param_name}\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Trial':<7} {'Value':<15} {'Slot F1':>10} {'Intent Acc':>12}  Status\n")
        f.write("-" * 70 + "\n")
        for r in results:   # keep chronological order in the table
            val_str = str(r["param_value"])
            f1_str = f"{r['metric']:.4f}" if r["metric"] != -1.0 else "  N/A"
            acc_str = f"{r['extras'].get('intent_acc', float('nan')):.4f}" \
                      if isinstance(r.get("extras"), dict) and "intent_acc" in r["extras"] \
                      else "     N/A"
            f.write(f"{r['trial']:<7} {val_str:<15} {f1_str:>10} {acc_str:>12}  {r['status']}\n")
        f.write("\n")
        if successful_results:
            best = successful_results[0]
            f.write(f"BEST  →  {param_name}={best['param_value']}  |  "
                    f"Slot F1={best['metric']:.4f}")
            if isinstance(best.get("extras"), dict) and "intent_acc" in best["extras"]:
                f.write(f"  |  Intent Acc={best['extras']['intent_acc']:.4f}")
            f.write("\n")
        else:
            f.write("No successful trials.\n")

    print(f"[Grid] Trial results written to: {trial_log_path}")

    return {
        "best_config": {param_name: successful_results[0]["param_value"]} if successful_results else {},
        "best_metric": successful_results[0]["metric"] if successful_results else -1.0,
        "best_extras": successful_results[0]["extras"] if successful_results else {},
        "results": results_sorted
    }

def sequential_grid_search(param_tuning_order: List[Dict[str, Any]],
                           base_config: Dict[str, Any],
                           run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                           base_results_dir: str = "bin/grid_search",
                           final_model_scores: Dict[str, Any] = None) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_results_dir = os.path.join(base_results_dir, f"sequential__{timestamp}")
    os.makedirs(parent_results_dir, exist_ok=True)
    
    current_best_config = copy.deepcopy(base_config)
    all_search_results = {}
    
    for step, param_cfg in enumerate(param_tuning_order, 1):
        param_name = param_cfg["name"]
        param_values = param_cfg["values"]
        
        print(f"\n[Sequential] Step {step}/{len(param_tuning_order)}: Tuning {param_name}")
        
        param_results_dir = os.path.join(parent_results_dir, f"{step:02d}_{param_name}")
        
        search_result = grid_search(
            param_name=param_name,
            param_values=param_values,
            base_config=current_best_config,
            run_fn=run_fn,
            results_dir=param_results_dir
        )
        
        all_search_results[param_name] = search_result
        
        if search_result["best_config"]:
            current_best_config.update(search_result["best_config"])
            print(f"[Sequential] Fixed {param_name}={search_result['best_config'][param_name]} " 
                  f"(F1: {search_result['best_metric']:.4f})")
    
    _generate_sequential_summary(parent_results_dir, all_search_results, current_best_config,
                                 final_model_scores=final_model_scores)
    return {"best_config": current_best_config, "all_results": all_search_results, "results_dir": parent_results_dir}

def _generate_sequential_summary(parent_dir, all_results, final_config, final_model_scores: Dict[str, Any] = None):
    summary_path = os.path.join(parent_dir, "sequential_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\nSEQUENTIAL GRID SEARCH SUMMARY\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("FINAL BEST CONFIGURATION\n" + "-" * 80 + "\n")
        for key, val in final_config.items():
            f.write(f"  {key}: {val}\n")

        f.write("\n\nSEARCH PROGRESSION\n" + "-" * 80 + "\n")
        for step, (param_name, result) in enumerate(all_results.items(), 1):
            f.write(f"\nStep {step}: {param_name}\n")
            f.write(f"  Best Value : {result['best_config'].get(param_name, 'N/A')}\n")
            f.write(f"  Best Slot F1: {result['best_metric']:.6f}\n")
            best_extras = result.get("best_extras", {})
            if isinstance(best_extras, dict) and "intent_acc" in best_extras:
                f.write(f"  Best Intent Acc: {best_extras['intent_acc']:.6f}\n")

            # Per-trial breakdown for this parameter
            trials = result.get("results", [])
            if trials:
                f.write(f"\n  {'Trial':<7} {'Value':<15} {'Slot F1':>10} {'Intent Acc':>12}  Status\n")
                f.write("  " + "-" * 60 + "\n")
                for r in sorted(trials, key=lambda x: x["trial"]):
                    val_str = str(r["param_value"])
                    f1_str = f"{r['metric']:.4f}" if r["metric"] != -1.0 else "  N/A"
                    acc_str = f"{r['extras'].get('intent_acc', float('nan')):.4f}" \
                              if isinstance(r.get("extras"), dict) and "intent_acc" in r["extras"] \
                              else "     N/A"
                    f.write(f"  {r['trial']:<7} {val_str:<15} {f1_str:>10} {acc_str:>12}  {r['status']}\n")

        if final_model_scores:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("BEST MODEL — FINAL EVALUATION\n")
            f.write("-" * 80 + "\n")
            score_labels = {
                "best_dev_f1":     "Best Dev Slot F1",
                "test_slot_f1":    "Test Slot F1",
                "test_intent_acc": "Test Intent Accuracy",
                "dev_intent_acc":  "Dev Intent Accuracy",
            }
            for key, label in score_labels.items():
                if key in final_model_scores:
                    f.write(f"  {label:<26}: {final_model_scores[key]:.4f}\n")

    print(f"[Grid] Sequential summary written to: {summary_path}")