import sys
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader

# Import modular project blocks
from utils import prepare_atis_data, IntentsAndSlots, collate_fn_atis, PAD_TOKEN
from model import ModelIAS
from functions import set_seed, init_weights, train_loop, eval_loop, save_experiment, sequential_grid_search, append_final_scores_to_summary

def main():
    # Anchor all relative paths (bin/, dataset/, config.json) to the script's own directory,
    # so the script runs correctly regardless of the working directory it is launched from.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 0. Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 1. Command-Line Argument Interface Setup
    parser = argparse.ArgumentParser(description="Joint Intent Classification and Slot Filling Model.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, directly run evaluation.")
    parser.add_argument("--model_path", type=str, default=f"bin/{config['experiment_name']}/{config['experiment_name']}.pt")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter grid search.")
    args = parser.parse_args()

    # Normalize path separators so Windows-style backslash paths work on Linux (e.g. Colab).
    args.model_path = args.model_path.replace("\\", "/")

    # If running in evaluation mode, load the config saved alongside the checkpoint so that
    # the model is reconstructed with the exact architecture it was trained with.
    if args.eval_only:
        checkpoint_dir = os.path.dirname(args.model_path)
        checkpoint_config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(checkpoint_config_path):
            with open(checkpoint_config_path, 'r') as f:
                config.update(json.load(f))
            print(f"[Config] Loaded evaluation configuration from {checkpoint_config_path}")
        else:
            print(f"[Warning] No checkpoint config found at {checkpoint_config_path}. Falling back to root config.")

    print(f"[Config] Experiment: {config['experiment_name']} | Model: {config.get('model_type', 'N/A')} | Optimizer: {config.get('optimizer', 'N/A')}")

    # 2. Seeding & Hardware Configuration
    set_seed(1234)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Initializing pipeline on execution device: {device}")

    # 3. Dataset Loading (Mapping over to ATIS)
    train_raw, dev_raw, test_raw, lang = prepare_atis_data()
    
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=partial(collate_fn_atis, device=device), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=partial(collate_fn_atis, device=device))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=partial(collate_fn_atis, device=device))

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    print(f"[Dataset] Vocabulary: {vocab_len} | Intents: {out_int} | Slots: {out_slot}")

    # Core Configuration Parameters
    experiment_name = config['experiment_name']
    emb_size = config.get('emb_size', 300)
    hid_size = config.get('hidden_size', 200)
    lr = config.get('lr', 0.0001)
    clip = config.get('clip', 5)
    n_epochs = config.get('n_epochs', 200)
    patience = config.get('patience', 3)
    bidirectional = config.get('bidirectional', True)
    config['bidirectional'] = bidirectional

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    # ================= EVALUATION ONLY MODE =================
    if args.eval_only:
        if not os.path.exists(args.model_path):
            print(f"\n[Error] Checkpoint not found at: {args.model_path}")
            sys.exit(1)
            
        print(f"\n=== Evaluation Mode ===")
        model_kwargs = {
            "hid_size":     hid_size,
            "out_slot":     out_slot,
            "out_int":      out_int,
            "emb_size":     emb_size,
            "vocab_len":    vocab_len,
            "pad_index":      PAD_TOKEN,
            "dropout_rate":   config.get('dropout_rate', 0.1),
            "bidirectional":  config.get('bidirectional', True),
        }
        
        model = ModelIAS(**model_kwargs).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        
        results_dev, intent_dev, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        dev_avg = (results_dev['total']['f'] + intent_dev['accuracy']) / 2.0
        test_avg = (results_test['total']['f'] + intent_test['accuracy']) / 2.0

        print(f"\n{'='*48}")
        print(f"Validation F1: {results_dev['total']['f']:.4f} | Intent Acc: {intent_dev['accuracy']:.4f} | Average: {dev_avg:.4f}")
        print(f"Test Set F1:   {results_test['total']['f']:.4f} | Intent Acc: {intent_test['accuracy']:.4f} | Average: {test_avg:.4f}")
        return

    # ================= TUNING MODE =================
    if args.tune:
        tuning_grid = config.get('tuning_grid', {})

        # The search order is exactly the key order of config.json's "tuning_grid" —
        # Python dicts (and json.load) preserve insertion order, so whatever order the
        # parameters are written in the JSON file is the order they're tuned in. Each
        # parameter is fixed at its best value before the next one is searched, so put
        # whichever parameter most shapes the loss landscape first (e.g. lr), and the
        # least impactful one last (e.g. clip) — reorder the JSON to change this.
        param_tuning_order = [
            {"name": name, "values": values}
            for name, values in tuning_grid.items()
        ]

        if not param_tuning_order:
            print("[Tune] No parameters found in config 'tuning_grid'. Exiting.")
            return

        print(f"[Tune] Will search {len(param_tuning_order)} parameter(s) sequentially: "
              f"{[p['name'] for p in param_tuning_order]}")

        def run_single_trial(trial_cfg):
            set_seed(1234)
            model_kwargs = {
                "hid_size":      trial_cfg.get('hidden_size',   hid_size),
                "out_slot":      out_slot,
                "out_int":       out_int,
                "emb_size":      trial_cfg.get('emb_size',      emb_size),
                "vocab_len":     vocab_len,
                "pad_index":     PAD_TOKEN,
                "dropout_rate":  trial_cfg.get('dropout_rate',  config.get('dropout_rate', 0.1)),
                "bidirectional": trial_cfg.get('bidirectional', bidirectional),
            }

            model_local = ModelIAS(**model_kwargs).to(device)
            model_local.apply(init_weights)
            optimizer_local = optim.Adam(model_local.parameters(), lr=trial_cfg.get('lr', lr))

            best_f1_local = 0
            best_state_local = None
            current_pat = trial_cfg.get('patience', patience)
            
            for epoch_local in range(1, trial_cfg.get('n_epochs', n_epochs) + 1):
                train_loop(train_loader, optimizer_local, criterion_slots, criterion_intents,
                           model_local, trial_cfg.get('clip', clip))
                
                if epoch_local % 5 == 0:
                    res_dev, _, _ = eval_loop(dev_loader, criterion_slots, criterion_intents,
                                              model_local, lang)
                    f1_dev = res_dev['total']['f']
                    
                    if f1_dev > best_f1_local:
                        best_f1_local = f1_dev
                        best_state_local = {k: v.cpu() for k, v in model_local.state_dict().items()}
                        current_pat = trial_cfg.get('patience', patience)
                    else:
                        current_pat -= 1
                        
                    if current_pat <= 0:
                        break

            if best_state_local is None:
                return -1.0, {}

            best_model_local = ModelIAS(**model_kwargs).to(device)
            best_model_local.load_state_dict(best_state_local)
            final_res, final_intent, _ = eval_loop(test_loader, criterion_slots, criterion_intents,
                                                   best_model_local, lang)
            slot_f1    = final_res['total']['f']
            intent_acc = final_intent['accuracy']
            avg_metric = (slot_f1 + intent_acc) / 2.0   # ranking metric
            return avg_metric, {
                "slot_f1":     slot_f1,
                "intent_acc":  intent_acc,
                "best_dev_f1": best_f1_local,
            }

        search_results = sequential_grid_search(
            param_tuning_order=param_tuning_order,
            base_config=config,
            run_fn=run_single_trial,
            base_results_dir=os.path.join("bin", config.get('experiment_name', 'grid_search')),
        )
        best_cfg = search_results['best_config']
        print(f"\n=== GRID SEARCH COMPLETE ===\nFinal Best Config: {best_cfg}\n")

        # Pull the definitive scores from the last search step's best trial —
        # that trial was evaluated with the fully-fixed config, so its numbers are
        # the most representative. The stored metric is already the average.
        last_step_result = list(search_results['all_results'].values())[-1]
        best_extras = last_step_result.get('best_extras', {})
        final_model_scores = {
            k: best_extras[k]
            for k in ('slot_f1', 'intent_acc', 'best_dev_f1')
            if k in best_extras
        }
        if 'slot_f1' in best_extras and 'intent_acc' in best_extras:
            final_model_scores['avg_metric'] = (best_extras['slot_f1'] + best_extras['intent_acc']) / 2.0

        print(f"{'='*48}")
        print(f"Test Slot F1:     {final_model_scores.get('slot_f1', float('nan')):.4f}")
        print(f"Test Intent Acc:  {final_model_scores.get('intent_acc', float('nan')):.4f}")
        print(f"Average (F1+Acc): {final_model_scores.get('avg_metric', float('nan')):.4f}")
        print(f"{'='*48}")

        append_final_scores_to_summary(search_results['results_dir'], final_model_scores)
        return

    # ================= STANDARD TRAINING MODE =================
    print("\n=== Training & Optimization Mode ===")
    
    model_kwargs = {
        "hid_size":     hid_size,
        "out_slot":     out_slot,
        "out_int":      out_int,
        "emb_size":     emb_size,
        "vocab_len":    vocab_len,
        "pad_index":    PAD_TOKEN,
        "dropout_rate": config.get('dropout_rate', 0.1),
        "bidirectional": bidirectional,
    }

    model = ModelIAS(**model_kwargs).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_train, losses_dev = [], []
    best_f1, best_model_state = 0, None
    current_patience = patience 
    stopped_at_epoch = n_epochs  # will be updated if early stopping fires

    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch Progress")
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        train_mean_loss = np.asarray(loss).mean() if loss else 0.0
        losses_train.append(train_mean_loss)
        
        # Evaluate dev every 5 epochs to reduce overhead during the long training run
        if epoch % 5 == 0:
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            dev_mean_loss = np.asarray(loss_dev).mean() if loss_dev else 0.0
            losses_dev.append(dev_mean_loss)

            f1 = results_dev['total']['f']
            pbar.set_description(f"Epoch {epoch} | Slot F1: {f1:.4f} | Intent Acc: {intent_res['accuracy']:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_dev_intent_acc = intent_res['accuracy']
                current_patience = patience
            else:
                current_patience -= 1
                
            if current_patience <= 0:
                stopped_at_epoch = epoch
                print(f"\n[Early Stopping] Triggered at Epoch {epoch} due to plateauing validation F1.")
                break 

    best_model = ModelIAS(**model_kwargs).to(device)
    if best_model_state: best_model.load_state_dict(best_model_state)

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
    avg_metric = (results_test['total']['f'] + intent_test['accuracy']) / 2.0
    print(f"\n{'='*48}")
    print(f"Optimal Configuration Model Evaluation complete!")
    print(f"Best Dev F1 Achieved: {best_f1:.4f}")
    print(f"Slot F1 (Test):       {results_test['total']['f']:.4f}")
    print(f"Intent Accuracy:      {intent_test['accuracy']:.4f}")
    print(f"Average (F1+Acc):     {avg_metric:.4f}")
    print(f"{'='*48}")

    config['best_dev_f1'] = best_f1
    config['final_test_f1'] = results_test['total']['f']
    config['final_intent_acc'] = intent_test['accuracy']
    config['avg_metric'] = avg_metric

    final_scores = {
        "best_dev_f1":     best_f1,
        "dev_intent_acc":  best_dev_intent_acc if 'best_dev_intent_acc' in dir() else None,
        "test_slot_f1":    results_test['total']['f'],
        "test_intent_acc": intent_test['accuracy'],
        "avg_metric":      avg_metric,
        "stopped_at_epoch": stopped_at_epoch,
    }
    # Remove None entries
    final_scores = {k: v for k, v in final_scores.items() if v is not None}
    
    save_experiment(best_model, config, losses_train, losses_dev, name=experiment_name,
                    final_scores=final_scores)

if __name__ == "__main__":
    main()