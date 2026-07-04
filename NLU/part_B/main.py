import sys
import os
# Guarantee that conll.py (downloaded next to this script) is importable
# regardless of what directory the user launches from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import numpy as np
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import prepare_bert_data, collate_fn
from model import JointBERT
from functions import (set_seed, train_loop, eval_loop, save_experiment,
                       sequential_grid_search, append_final_scores_to_summary)


def build_model(config, num_slots, num_intents, device):
    model = JointBERT(
        num_slots    = num_slots,
        num_intents  = num_intents,
        dropout_rate = config.get('dropout_rate', 0.1),
        model_name   = config.get('bert_model', 'bert-base-uncased'),
    ).to(device)
    return model


def build_loaders(train_dataset, dev_dataset, test_dataset, config, device):
    _collate = partial(collate_fn, device=device)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              collate_fn=_collate, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=config['batch_size'],
                              collate_fn=_collate)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              collate_fn=_collate)
    return train_loader, dev_loader, test_loader


def run_training(train_loader, dev_loader, config, model, optimizer,
                 criterion_slots, criterion_intents, lang):
    """
    Core training loop. Returns (best_state_dict, best_dev_f1, best_dev_intent_acc,
    train_losses, dev_losses, stopped_at_epoch).
    BERT is evaluated every epoch (not every 5) because fine-tuning converges fast.
    """
    n_epochs  = config.get('n_epochs', 10)
    patience  = config.get('patience', 3)
    clip      = config.get('clip', 1.0)

    best_joint      = 0.0       # (F1 + intent_acc) / 2
    best_state      = None
    best_dev_f1     = 0.0
    best_dev_acc    = 0.0
    patience_left   = patience
    stopped_epoch   = n_epochs
    train_losses    = []
    dev_losses      = []

    pbar = tqdm(range(1, n_epochs + 1), desc="Fine-tuning")
    for epoch in pbar:
        loss_train = train_loop(train_loader, optimizer, criterion_slots,
                                criterion_intents, model, clip=clip)
        train_mean = float(np.mean(loss_train))
        train_losses.append(train_mean)

        results_dev, intent_res, loss_dev = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang)
        dev_mean = float(np.mean(loss_dev))
        dev_losses.append(dev_mean)

        dev_f1  = results_dev['total']['f']
        dev_acc = intent_res['accuracy']
        # Joint metric mirrors the paper — average of both task metrics
        joint   = (dev_f1 + dev_acc) / 2.0

        pbar.set_description(
            f"Epoch {epoch:02d} | Train: {train_mean:.4f} | Dev: {dev_mean:.4f} "
            f"| Slot F1: {dev_f1:.4f} | Intent Acc: {dev_acc:.4f} | Joint: {joint:.4f}"
        )

        if joint > best_joint:
            best_joint   = joint
            best_dev_f1  = dev_f1
            best_dev_acc = dev_acc
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            stopped_epoch = epoch
            print(f"\n[Early Stopping] Epoch {epoch} | Best joint metric: {best_joint:.4f}")
            break

    return best_state, best_dev_f1, best_dev_acc, train_losses, dev_losses, stopped_epoch


def main():
    # Anchor all relative paths (bin/, dataset/, config.json) to the script's own directory,
    # so the script runs correctly regardless of the working directory it is launched from.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # ── Config ───────────────────────────────────────────────────────────
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # ── CLI ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Joint BERT Intent + Slot Fine-Tuning")
    parser.add_argument("--eval_only",   action="store_true",
                        help="Skip training; evaluate a saved checkpoint.")
    parser.add_argument("--model_path",  type=str,
                        default=f"bin/{config['experiment_name']}/{config['experiment_name']}.pt")
    parser.add_argument("--tune",        action="store_true",
                        help="Run sequential hyperparameter grid search.")
    args = parser.parse_args()

    # Normalize path separators so Windows-style backslash paths work on Linux (e.g. Colab).
    args.model_path = args.model_path.replace("\\", "/")

    # If running in evaluation mode, load the config saved alongside the checkpoint BEFORE
    # data loading — bert_model drives the tokenizer, so it must be correct from the start.
    if args.eval_only:
        checkpoint_dir = os.path.dirname(args.model_path)
        checkpoint_config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(checkpoint_config_path):
            with open(checkpoint_config_path) as f:
                config.update(json.load(f))
            print(f"[Config] Loaded evaluation configuration from {checkpoint_config_path}")
        else:
            print(f"[Warning] No checkpoint config found at {checkpoint_config_path}. Falling back to root config.")

    print(f"[Config] Experiment : {config['experiment_name']}")
    print(f"[Config] BERT model : {config.get('bert_model', 'bert-base-uncased')}")

    # ── Seed & device ────────────────────────────────────────────────────
    set_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_dataset, dev_dataset, test_dataset, lang, _ = prepare_bert_data(
        bert_model_name=config.get('bert_model', 'bert-base-uncased')
    )
    train_loader, dev_loader, test_loader = build_loaders(
        train_dataset, dev_dataset, test_dataset, config, device
    )

    num_slots   = len(lang.slot2id)
    num_intents = len(lang.intent2id)

    criterion_slots   = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_intents = nn.CrossEntropyLoss()

    # ── Eval-only mode ───────────────────────────────────────────────────
    if args.eval_only:
        if not os.path.exists(args.model_path):
            print(f"[Error] Checkpoint not found: {args.model_path}")
            sys.exit(1)

        model = build_model(config, num_slots, num_intents, device)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

        results_dev,  intent_dev,  _ = eval_loop(dev_loader,  criterion_slots, criterion_intents, model, lang)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)

        print(f"\n{'='*52}")
        print(f"Dev  — Slot F1: {results_dev['total']['f']:.4f}  | Intent Acc: {intent_dev['accuracy']:.4f}")
        print(f"Test — Slot F1: {results_test['total']['f']:.4f}  | Intent Acc: {intent_test['accuracy']:.4f}")
        print(f"{'='*52}")
        return

    # ── Tuning mode ──────────────────────────────────────────────────────
    if args.tune:
        tuning_grid = config.get('tuning_grid', {})

        # Order: lr first (most critical for fine-tuning), weight_decay second
        # (L2 regularisation interacts directly with lr), then dropout, batch
        # size, and finally clipping (least impactful).
        _TUNE_ORDER = ["lr", "weight_decay", "dropout_rate", "batch_size", "clip"]
        param_tuning_order = [
            {"name": p, "values": tuning_grid[p]}
            for p in _TUNE_ORDER if p in tuning_grid
        ]

        if not param_tuning_order:
            print("[Tune] No parameters in config 'tuning_grid'. Exiting.")
            return

        print(f"[Tune] Searching: {[p['name'] for p in param_tuning_order]}")

        def run_single_trial(trial_cfg):
            set_seed(1234)

            # Rebuild loaders if batch_size changed
            t_loader, d_loader, _ = build_loaders(
                train_dataset, dev_dataset, test_dataset, trial_cfg, device
            )
            trial_model = build_model(trial_cfg, num_slots, num_intents, device)
            trial_opt   = optim.AdamW(
                trial_model.parameters(),
                lr=trial_cfg.get('lr', config['lr']),
                weight_decay=trial_cfg.get('weight_decay', config.get('weight_decay', 0.01))
            )

            best_state, best_f1, best_acc, _, _, _ = run_training(
                t_loader, d_loader, trial_cfg, trial_model,
                trial_opt, criterion_slots, criterion_intents, lang
            )

            if best_state is None:
                return -1.0, {}

            # Evaluate on test with best checkpoint
            trial_model.load_state_dict(best_state)
            res_test, intent_test, _ = eval_loop(
                test_loader, criterion_slots, criterion_intents, trial_model, lang
            )
            return res_test['total']['f'], {
                "intent_acc":  intent_test['accuracy'],
                "best_dev_f1": best_f1,
            }

        search_results = sequential_grid_search(
            param_tuning_order=param_tuning_order,
            base_config=config,
            run_fn=run_single_trial,
            base_results_dir=os.path.join("bin", config['experiment_name']),
        )
        best_cfg = search_results['best_config']
        print(f"\n=== GRID SEARCH COMPLETE ===\nBest config: {best_cfg}")

        # Collect final scores from the last search step (no re-run needed)
        final_scores = {}
        for result in search_results['all_results'].values():
            if result['best_metric'] != -1.0:
                final_scores['test_slot_f1']    = result['best_metric']
                final_scores['test_intent_acc'] = result['best_extras'].get('intent_acc', 0)
        if 'test_slot_f1' in final_scores and 'test_intent_acc' in final_scores:
            final_scores['avg_metric'] = (final_scores['test_slot_f1'] + final_scores['test_intent_acc']) / 2.0

        print(f"{'='*52}")
        print(f"Test Slot F1:     {final_scores.get('test_slot_f1', float('nan')):.4f}")
        print(f"Test Intent Acc:  {final_scores.get('test_intent_acc', float('nan')):.4f}")
        print(f"Average (F1+Acc): {final_scores.get('avg_metric', float('nan')):.4f}")
        print(f"{'='*52}")

        append_final_scores_to_summary(search_results['results_dir'], final_scores)
        return

    # ── Standard training mode ───────────────────────────────────────────
    print("\n=== Fine-Tuning JointBERT ===")

    model     = build_model(config, num_slots, num_intents, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 3e-5),
        weight_decay=config.get('weight_decay', 0.01)
    )

    best_state, best_dev_f1, best_dev_acc, train_losses, dev_losses, stopped_epoch = run_training(
        train_loader, dev_loader, config, model,
        optimizer, criterion_slots, criterion_intents, lang
    )

    # Restore best checkpoint and evaluate on test
    if best_state:
        model.load_state_dict(best_state)
    results_test, intent_test, _ = eval_loop(
        test_loader, criterion_slots, criterion_intents, model, lang
    )

    avg_metric = (results_test['total']['f'] + intent_test['accuracy']) / 2.0
    print(f"\n{'='*52}")
    print(f"Best Dev — Slot F1: {best_dev_f1:.4f} | Intent Acc: {best_dev_acc:.4f}")
    print(f"Test     — Slot F1: {results_test['total']['f']:.4f} | Intent Acc: {intent_test['accuracy']:.4f}")
    print(f"Average (F1+Acc)  : {avg_metric:.4f}")
    print(f"{'='*52}")

    final_scores = {
        "best_dev_f1":      best_dev_f1,
        "dev_intent_acc":   best_dev_acc,
        "test_slot_f1":     results_test['total']['f'],
        "test_intent_acc":  intent_test['accuracy'],
        "avg_metric":       avg_metric,
        "stopped_at_epoch": stopped_epoch,
    }
    config.update({k: v for k, v in final_scores.items()})

    save_experiment(model, config, train_losses, dev_losses,
                    name=config['experiment_name'], final_scores=final_scores)


if __name__ == "__main__":
    main()