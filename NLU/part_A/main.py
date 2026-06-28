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

# Import modular project blocks
from utils import get_atis_dataloaders
from model import ModelIAS
from functions import set_seed, init_weights, train_loop, eval_loop, save_experiment, sequential_grid_search

def main():
    # 0. Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"[Config] Loaded configuration from {config_path}")
    print(f"[Config] Experiment: {config['experiment_name']} | Model: {config['model_type']} | Optimizer: {config['optimizer']}")

    # 1. Command-Line Argument Interface Setup
    parser = argparse.ArgumentParser(description="Joint Intent Classification and Slot Filling Model.")
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
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter grid search using 'tuning_grid' in config.json or a default grid."
    )
    args = parser.parse_args()

    # 2. Seeding & Hardware Configuration
    set_seed(1234)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[System] Initializing pipeline on execution device: {device}")

    # 3. Dataset Loading (Mapping over to ATIS)
    pad_idx = 0 # Default Pad Index
    train_loader, dev_loader, test_loader, lang = get_atis_dataloaders(config['batch_size'], pad_idx, device)
    
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    print(f"[Dataset] Vocabulary: {vocab_len} | Intents: {out_int} | Slots: {out_slot}")

    # Core Configuration Parameters
    experiment_name = config['experiment_name']
    optimizer_name = config['optimizer']
    emb_size = config.get('emb_size', 300)
    hid_size = config.get('hidden_size', 200)
    lr = config.get('lr', 0.0001)
    clip = config.get('clip', 5)
    n_epochs = config.get('n_epochs', 200)
    batch_size = config.get('batch_size', 64)
    patience = config.get('patience', 3)
    
    # New Hyperparameters for Part 2.A (Dropout & Bidirectionality)
    dropout_val = config.get('dropout', 0.1)
    bidirectional = config.get('bidirectional', False)

    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_intents = nn.CrossEntropyLoss() # No PAD token for intents

    # ================= EVALUATION ONLY MODE =================
    if args.eval_only:
        if not os.path.exists(args.model_path):
            print(f"\n[Error] Saved model checkpoint not found at: {args.model_path}")
            print("Verify checkpoint path or execute standard training first.")
            sys.exit(1)
            
        print(f"\n=== Evaluation Mode ===")
        print(f"[Load] Loading model parameters from: {args.model_path}")
        
        model_kwargs = {
            "hid_size": hid_size, "out_slot": out_slot, "out_int": out_int,
            "emb_size": emb_size, "vocab_len": vocab_len, "pad_index": pad_idx
        }
        
        model = ModelIAS(**model_kwargs).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        results_dev, intent_dev, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        
        print(f"\n{'='*48}")
        print(f"Pre-Trained Model Evaluation Metrics:")
        print(f"Validation F1: {results_dev['total']['f']:.4f} | Intent Acc: {intent_dev['accuracy']:.4f}")
        print(f"Test Set F1:   {results_test['total']['f']:.4f} | Intent Acc: {intent_test['accuracy']:.4f}")
        print(f"{'='*48}")
        return

    # ================= TUNING MODE =================
    if args.tune:
        tuning_grid = config.get('tuning_grid', {})
        
        param_tuning_order = [
            # {
            #     "name": "batch_size",
            #     # they were almost the same from ???.4 to ???.3 so i decided to go with what i had before which is ???
            #     "values": tuning_grid.get("batch_size", [16, 32, 64, 128])
            # },
            # {
            #     "name": "hidden_size",
            #     # ??? was the best so we moved from 400 to 200
            #     "values": tuning_grid.get("hidden_size", [100, 200, 300, 400])
            # },
            # {
            #     "name": "lr",
            #     # 10 was the best so we changed from 5 to 10
            #     # "values": tuning_grid.get("lr", [0.01, 0.05, 0.1, 0.5, 1, 5])
            #     "values": tuning_grid.get("lr", [5, 10, 50])
            # },
            # {
            #     "name": "emb_size",
            #     # ??? was the best so we moved from 400 to 200
            #     "values": tuning_grid.get("emb_size", [100, 200, 300, 400])
            # },
            # {
            #     "name": "clip",
            #     # ??? was the best so we moved from ??? to ???
            #     "values": tuning_grid.get("clip", [0.1, 0.5, 1, 3, 5, 10, 50])
            # },
            # {
            #     "name": "dropout",
            #     "values": tuning_grid.get("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
            # },
            {
                "name": "lr",
                "values": tuning_grid.get("lr", [0.0001, 0.001, 0.01])
            },
        ]

        def run_single_trial(trial_cfg):
            set_seed(1234)
            model_kwargs = {
                "hid_size": trial_cfg.get('hidden_size', hid_size),
                "out_slot": out_slot, "out_int": out_int,
                "emb_size": trial_cfg.get('emb_size', emb_size),
                "vocab_len": vocab_len, "pad_index": pad_idx
            }

            model_local = ModelIAS(**model_kwargs).to(device)
            model_local.apply(init_weights)

            lr_trial = trial_cfg.get('lr', lr)
            optimizer_local = optim.Adam(model_local.parameters(), lr=lr_trial)

            best_f1_local = 0
            best_state_local = None
            current_pat = trial_cfg.get('patience', patience)
            
            for epoch_local in range(1, trial_cfg.get('n_epochs', n_epochs) + 1):
                train_loop(train_loader, optimizer_local, criterion_slots, criterion_intents, model_local, clip)
                
                # Check performance every 5 epochs as specified
                if epoch_local % 5 == 0:
                    res_dev, intent_res, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model_local, lang)
                    f1_dev = res_dev['total']['f']
                    
                    if f1_dev > best_f1_local:
                        best_f1_local = f1_dev
                        best_state_local = {k: v.cpu() for k, v in model_local.state_dict().items()}
                        current_pat = trial_cfg.get('patience', patience)
                    else:
                        current_pat -= 1
                        
                    if current_pat <= 0:
                        break

            # If model didn't train properly
            if best_state_local is None: return -1.0, {}

            best_model_local = ModelIAS(**model_kwargs).to(device)
            best_model_local.load_state_dict(best_state_local)
            final_res, final_intent, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model_local, lang)

            return final_res['total']['f'], {"intent_acc": final_intent['accuracy']}

        search_results = sequential_grid_search(
            param_tuning_order=param_tuning_order,
            base_config=config,
            run_fn=run_single_trial,
            base_results_dir=os.path.join("bin", config.get('experiment_name', 'grid_search'))
        )
        print("\n" + "="*70)
        print("SEQUENTIAL GRID SEARCH COMPLETE")
        print("="*70)
        print(f"Final Best Configuration: {search_results['best_config']}")
        print("="*70 + "\n")
        return

    # ================= STANDARD TRAINING MODE =================
    print("\n=== Training & Optimization Mode ===")
    
    model_kwargs = {
        "hid_size": hid_size, "out_slot": out_slot, "out_int": out_int,
        "emb_size": emb_size, "vocab_len": vocab_len, "pad_index": pad_idx
    }
    
    model = ModelIAS(**model_kwargs).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_train = []
    losses_dev = []
    
    best_f1 = 0
    best_model_state = None
    current_patience = patience 

    # As requested: Training loop that checks performance every 5 epochs
    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch Progress")
    for epoch in pbar:
        # Run training epoch
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        losses_train.append(np.asarray(loss).mean())
        
        # We check the performance every 5 epochs
        if epoch % 5 == 0:
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev['total']['f']
            pbar.set_description(f"Epoch {epoch} | Slot F1: {f1:.4f} | Intent Acc: {intent_res['accuracy']:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                current_patience = patience
            else:
                current_patience -= 1
                
            if current_patience <= 0:
                print(f"\n[Early Stopping] Triggered at Epoch {epoch} due to plateauing validation F1.")
                break 

    # Restore the best performing parameters from the training run
    best_model = ModelIAS(**model_kwargs).to(device)
    if best_model_state:
        best_model.load_state_dict(best_model_state)

    # Final evaluate run across testing dataset
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
    print(f"\n{'='*48}")
    print(f"Optimal Configuration Model Evaluation complete!")
    print(f"Best Dev F1 Achieved: {best_f1:.4f}")
    print(f"Slot F1 (Test):       {results_test['total']['f']:.4f}")
    print(f"Intent Accuracy:      {intent_test['accuracy']:.4f}")
    print(f"{'='*48}")

    config_details = {
        "experiment_name": experiment_name,
        "model_type": config.get('model_type', 'ModelIAS'),
        "optimizer": optimizer_name,
        "emb_size": emb_size,
        "hidden_size": hid_size,
        "lr": lr,
        "patience": patience,
        "batch_size": batch_size,
        "clip": clip,
        "n_epochs": n_epochs,
        "dropout": dropout_val,
        "bidirectional": bidirectional,
        "best_dev_f1": best_f1,
        "final_test_f1": results_test['total']['f'],
        "final_intent_acc": intent_test['accuracy']
    }
    
    save_experiment(
        model=best_model,
        hyperparameters=config_details,
        train_losses=losses_train,
        dev_metrics=losses_dev,
        name=experiment_name
    )

if __name__ == "__main__":
    main()