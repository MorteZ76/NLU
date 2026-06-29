import os
import copy
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import classification_report

try:
    from conll import evaluate
except ImportError:
    pass


# =========================================================================
# REPRODUCIBILITY
# =========================================================================

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep deterministic=False for BERT — CuDNN non-determinism is acceptable
    # and deterministic mode causes significant slowdowns with transformers.
    torch.backends.cudnn.benchmark = True
    print(f"[Init] Seed set to {seed}")


# =========================================================================
# TRAIN / EVAL LOOPS
# =========================================================================

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=1.0):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()

        slot_logits, intent_logits = model(sample['input_ids'], sample['attention_mask'])

        # slot_ids has -100 at [CLS]/[SEP]/padding/continuation sub-tokens
        # CrossEntropyLoss with ignore_index=-100 handles this automatically
        loss_slot   = criterion_slots(slot_logits, sample['slot_ids'])
        loss_intent = criterion_intents(intent_logits, sample['intent_ids'])
        loss        = loss_slot + loss_intent

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss_array.append(loss.item())

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array  = []
    ref_intents = []
    hyp_intents = []
    ref_slots   = []
    hyp_slots   = []

    with torch.no_grad():
        for sample in data:
            slot_logits, intent_logits = model(sample['input_ids'], sample['attention_mask'])

            loss_slot   = criterion_slots(slot_logits, sample['slot_ids'])
            loss_intent = criterion_intents(intent_logits, sample['intent_ids'])
            loss_array.append((loss_slot + loss_intent).item())

            # Intent predictions
            pred_intent_ids = torch.argmax(intent_logits, dim=1).tolist()
            ref_intents.extend([lang.id2intent[i] for i in sample['intent_ids'].tolist()])
            hyp_intents.extend([lang.id2intent[i] for i in pred_intent_ids])

            # Slot predictions — reconstruct at WORD level, not sub-token level
            pred_slot_ids = torch.argmax(slot_logits, dim=1)   # [B, seq_len]

            for i, raw_item in enumerate(sample['raw_batch']):
                words      = raw_item['words']
                gold_ids   = sample['slot_ids'][i].tolist()
                pred_ids   = pred_slot_ids[i].tolist()

                # Positions where gold != -100 are the first sub-tokens of real words
                gold_labels = []
                pred_labels = []
                for gold_id, pred_id in zip(gold_ids, pred_ids):
                    if gold_id != -100:
                        gold_labels.append(lang.id2slot[gold_id])
                        pred_labels.append(lang.id2slot[pred_id])

                # Pair each word with its label for conll evaluation
                ref_slots.append(list(zip(words, gold_labels)))
                hyp_slots.append(list(zip(words, pred_labels)))

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print(f"[Eval] conll evaluate warning: {ex}")
        results = {"total": {"f": 0.0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


# =========================================================================
# SAVE EXPERIMENT
# =========================================================================

def save_experiment(model, hyperparameters, train_losses, dev_losses,
                    name="bert_atis", final_scores: Dict[str, Any] = None):
    os.makedirs("bin", exist_ok=True)
    exp_dir = os.path.join("bin", name)
    os.makedirs(exp_dir, exist_ok=True)

    model_path  = os.path.join(exp_dir, f"{name}.pt")
    config_path = os.path.join(exp_dir, "config.json")

    torch.save(model.state_dict(), model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hyperparameters, f, indent=4)

    # Loss plot
    if train_losses:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", color="#1f77b4", marker='o', linewidth=2)
        if dev_losses:
            plt.plot(dev_losses, label="Dev Loss", color="#ff7f0e", marker='s',
                     linestyle="--", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss — {name}", fontweight='bold')
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(exp_dir, "loss_plot.png"), dpi=200, bbox_inches='tight')
        plt.close()

    # Human-readable summary
    _DISPLAY_KEYS = ["model_type", "bert_model", "optimizer", "lr", "dropout_rate",
                     "clip", "n_epochs", "patience", "batch_size", "weight_decay"]
    summary_path = os.path.join(exp_dir, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EXPERIMENT RESULTS — {name}\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("HYPERPARAMETERS\n" + "-" * 60 + "\n")
        for key in _DISPLAY_KEYS:
            if key in hyperparameters:
                f.write(f"  {key:<20}: {hyperparameters[key]}\n")
        f.write("\n")

        if final_scores:
            f.write("FINAL SCORES\n" + "-" * 60 + "\n")
            labels = {
                "best_dev_f1":      "Best Dev Slot F1",
                "dev_intent_acc":   "Best Dev Intent Acc",
                "test_slot_f1":     "Test Slot F1",
                "test_intent_acc":  "Test Intent Accuracy",
                "stopped_at_epoch": "Stopped at Epoch",
            }
            for key, label in labels.items():
                if key in final_scores:
                    val = final_scores[key]
                    f.write(f"  {label:<26}: {val:.4f}\n" if isinstance(val, float)
                            else f"  {label:<26}: {val}\n")
            f.write("\n")

        f.write("FILES\n" + "-" * 60 + "\n")
        f.write(f"  Checkpoint : {model_path}\n")
        f.write(f"  Config     : {config_path}\n")
        if train_losses:
            f.write(f"  Loss plot  : {os.path.join(exp_dir, 'loss_plot.png')}\n")
        f.write(f"  This file  : {summary_path}\n")

    print(f"[Save] Results written to: {exp_dir}/")


# =========================================================================
# GRID SEARCH
# =========================================================================

def grid_search(param_name: str,
                param_values: List[Any],
                base_config: Dict[str, Any],
                run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                results_dir: str = "bin/grid_search") -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)
    results = []

    with tqdm(total=len(param_values), desc=f"Searching {param_name}", unit="trial") as pbar:
        for idx, param_val in enumerate(param_values, start=1):
            trial_cfg = copy.deepcopy(base_config)
            trial_cfg[param_name] = param_val

            try:
                metric, extras = run_fn(trial_cfg)
                status = "✓ PASS" if metric != -1.0 else "✗ FAIL"
            except Exception as e:
                print(f"\n[Grid] Trial {idx} failed: {str(e)[:120]}")
                metric, extras, status = -1.0, {"error": str(e)}, "✗ ERROR"

            results.append({
                "trial": idx, "param_value": param_val,
                "metric": metric, "extras": extras, "status": status,
            })
            pbar.update(1)

    results_sorted     = sorted(results, key=lambda r: r["metric"], reverse=True)
    successful_results = [r for r in results_sorted if r["metric"] != -1.0]

    # Write per-param trial log
    log_path = os.path.join(results_dir, "trial_results.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"GRID SEARCH — Parameter: {param_name}\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Trial':<7} {'Value':<15} {'Slot F1':>10} {'Intent Acc':>12}  Status\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f1_str  = f"{r['metric']:.4f}" if r["metric"] != -1.0 else "  N/A"
            acc_str = (f"{r['extras']['intent_acc']:.4f}"
                       if isinstance(r.get("extras"), dict) and "intent_acc" in r["extras"]
                       else "     N/A")
            f.write(f"{r['trial']:<7} {str(r['param_value']):<15} {f1_str:>10} {acc_str:>12}  {r['status']}\n")
        f.write("\n")
        if successful_results:
            best = successful_results[0]
            f.write(f"BEST → {param_name}={best['param_value']}  |  Slot F1={best['metric']:.4f}")
            if isinstance(best.get("extras"), dict) and "intent_acc" in best["extras"]:
                f.write(f"  |  Intent Acc={best['extras']['intent_acc']:.4f}")
            f.write("\n")
        else:
            f.write("No successful trials.\n")
    print(f"[Grid] Trial results written to: {log_path}")

    return {
        "best_config":  {param_name: successful_results[0]["param_value"]} if successful_results else {},
        "best_metric":  successful_results[0]["metric"]  if successful_results else -1.0,
        "best_extras":  successful_results[0]["extras"]  if successful_results else {},
        "results":      results_sorted,
    }


def sequential_grid_search(param_tuning_order: List[Dict[str, Any]],
                            base_config: Dict[str, Any],
                            run_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
                            base_results_dir: str = "bin/grid_search") -> Dict[str, Any]:
    timestamp         = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir        = os.path.join(base_results_dir, f"sequential__{timestamp}")
    os.makedirs(parent_dir, exist_ok=True)

    current_best_config = copy.deepcopy(base_config)
    all_search_results  = {}

    for step, param_cfg in enumerate(param_tuning_order, 1):
        param_name   = param_cfg["name"]
        param_values = param_cfg["values"]
        print(f"\n[Sequential] Step {step}/{len(param_tuning_order)}: Tuning {param_name}")

        result = grid_search(
            param_name=param_name,
            param_values=param_values,
            base_config=current_best_config,
            run_fn=run_fn,
            results_dir=os.path.join(parent_dir, f"{step:02d}_{param_name}"),
        )
        all_search_results[param_name] = result

        if result["best_config"]:
            current_best_config.update(result["best_config"])
            print(f"[Sequential] Fixed {param_name}={result['best_config'][param_name]} "
                  f"(F1: {result['best_metric']:.4f})")

    _write_sequential_summary(parent_dir, all_search_results, current_best_config)
    return {
        "best_config": current_best_config,
        "all_results": all_search_results,
        "results_dir": parent_dir,
    }


def append_final_scores_to_summary(results_dir: str, final_model_scores: Dict[str, Any]):
    """Appends definitive test scores to an existing sequential_summary.txt."""
    summary_path = os.path.join(results_dir, "sequential_summary.txt")
    if not os.path.exists(summary_path):
        print(f"[Grid] Warning: summary not found at {summary_path}")
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("BEST MODEL — FINAL EVALUATION\n")
        f.write(f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 80 + "\n")
        labels = {
            "test_slot_f1":    "Test Slot F1",
            "test_intent_acc": "Test Intent Accuracy",
        }
        for key, label in labels.items():
            if key in final_model_scores:
                f.write(f"  {label:<26}: {final_model_scores[key]:.4f}\n")
    print(f"[Grid] Final scores appended to: {summary_path}")


def _write_sequential_summary(parent_dir, all_results, final_config):
    summary_path = os.path.join(parent_dir, "sequential_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\nSEQUENTIAL GRID SEARCH SUMMARY\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("FINAL BEST CONFIGURATION\n" + "-" * 80 + "\n")
        for key, val in final_config.items():
            if not isinstance(val, dict):   # skip nested tuning_grid
                f.write(f"  {key}: {val}\n")

        f.write("\n\nSEARCH PROGRESSION\n" + "-" * 80 + "\n")
        for step, (param_name, result) in enumerate(all_results.items(), 1):
            f.write(f"\nStep {step}: {param_name}\n")
            f.write(f"  Best Value  : {result['best_config'].get(param_name, 'N/A')}\n")
            f.write(f"  Best Slot F1: {result['best_metric']:.6f}\n")
            if isinstance(result.get("best_extras"), dict) and "intent_acc" in result["best_extras"]:
                f.write(f"  Best Intent Acc: {result['best_extras']['intent_acc']:.6f}\n")

            trials = result.get("results", [])
            if trials:
                f.write(f"\n  {'Trial':<7} {'Value':<15} {'Slot F1':>10} {'Intent Acc':>12}  Status\n")
                f.write("  " + "-" * 60 + "\n")
                for r in sorted(trials, key=lambda x: x["trial"]):
                    f1_str  = f"{r['metric']:.4f}" if r["metric"] != -1.0 else "  N/A"
                    acc_str = (f"{r['extras']['intent_acc']:.4f}"
                               if isinstance(r.get("extras"), dict) and "intent_acc" in r["extras"]
                               else "     N/A")
                    f.write(f"  {r['trial']:<7} {str(r['param_value']):<15} "
                            f"{f1_str:>10} {acc_str:>12}  {r['status']}\n")

    print(f"[Grid] Sequential summary written to: {summary_path}")