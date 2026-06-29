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
from functions import set_seed, init_weights, train_loop, eval_loop, save_experiment, sequential_grid_search, check_nt_asgd_trigger, swap_asgd_weights, restore_asgd_weights

def main():
    # Anchor all relative paths (bin/, dataset/, config.json) to the script's own directory,
    # so the script runs correctly regardless of the working directory it is launched from.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 0. Load default configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

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
        default=f"bin/{config.get('experiment_name', 'RNN')}/{config.get('experiment_name', 'RNN')}.pt", 
        help="Relative path to a pre-trained PyTorch weight checkpoint (.pt)."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter grid search using 'tuning_grid' in config.json or a default grid."
    )
    args = parser.parse_args()

    # Normalize path separators so Windows-style backslash paths work on Linux (e.g. Colab).
    args.model_path = args.model_path.replace("\\", "/")

    # ================= CONFIG OVERRIDE (EVAL ONLY) =================
    # If running in evaluation mode, prioritize the config stored with the model
    if args.eval_only:
        model_dir = os.path.dirname(args.model_path)
        eval_config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(eval_config_path):
            print(f"[Config] Loading evaluation configuration from {eval_config_path}")
            with open(eval_config_path, 'r') as f:
                config = json.load(f)
        else:
            print(f"[Warning] No config.json found at {eval_config_path}. Falling back to root config.")
            
    print(f"[Config] Experiment: {config.get('experiment_name')} | Model: {config.get('model_type')} | Optimizer: {config.get('optimizer')}")

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

    # Core Configuration Parameters
    experiment_name = config.get('experiment_name', 'default_exp')
    model_type = config.get('model_type', 'RNN')
    optimizer_name = config.get('optimizer', 'SGD')
    emb_size = config.get('emb_size', 200)
    hid_size = config.get('hidden_size', 200)
    lr = config.get('lr', 10)
    patience = config.get('patience', 3)
    batch_size = config.get('batch_size', 64)
    clip = config.get('clip', 0.1)
    n_epochs = config.get('n_epochs', 20)
    weight_tying = config.get('weight_tying', False)
    emb_drop = config.get('emb_drop', 0.1)
    out_drop = config.get('out_drop', 0.1)
    non_mono = config.get('non_mono', 5)

    # Initialize Validation and Test DataLoaders
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(collate_fn, pad_token=pad_idx, device=device)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(collate_fn, pad_token=pad_idx, device=device)
    )

    def resolve_model_sizes(emb_value, hidden_value):
        if weight_tying and emb_value != hidden_value:
            target_size = emb_value if emb_value is not None else hidden_value
            if hidden_value is None:
                hidden_value = target_size
            if emb_value is None:
                emb_value = target_size
            print(f"[Config] Weight tying enabled; aligning emb_size={target_size} and hidden_size={target_size}.")
            return target_size, target_size
        return emb_value, hidden_value

    emb_size, hid_size = resolve_model_sizes(emb_size, hid_size)

    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

    # ================= EVALUATION ONLY MODE =================
    if args.eval_only:
        if not os.path.exists(args.model_path):
            print(f"\n[Error] Saved model checkpoint not found at: {args.model_path}")
            print("Verify checkpoint path or execute standard training first.")
            sys.exit(1)
            
        print(f"\n=== Evaluation Mode ===")
        print(f"[Load] Loading model parameters from: {args.model_path}")
        
        # Instantiate architecture and load saved parameters
        model_kwargs = {
            "emb_size": emb_size, "hidden_size": hid_size, "output_size": vocab_len,
            "pad_index": pad_idx, "emb_dropout": emb_drop, "out_dropout": out_drop,
            "weight_tying": weight_tying
        }

        # Construct specific model architecture according to config
        if model_type.upper() == "LSTM":
            lstm_kwargs = {k: v for k, v in model_kwargs.items() if k != 'weight_tying'}
            model = LM_LSTM(**lstm_kwargs).to(device)
        else:
            model = LM_RNN(**model_kwargs).to(device)

        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

        # Re-establish weight tying after load_state_dict, which severs the shared
        # tensor reference even though values remain equal.
        if weight_tying and model_type.upper() != "LSTM":
            model.output.weight = model.embedding.weight
        
        # Calculate perplexity metrics across validation and test sets
        val_ppl, _ = eval_loop(dev_loader, criterion_eval, model)
        test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        
        print(f"\n{'='*48}")
        print(f"Pre-Trained Model Evaluation Metrics:")
        print(f"Validation Perplexity (PPL): {val_ppl:.3f}")
        print(f"Test Set Perplexity (PPL):   {test_ppl:.3f}")
        print(f"{'='*48}")

        # === Sample Sentence Generation ===
        print("\n=== Generating Sample Sentences ===")
        model.eval()
        temperature = 0.8  # Slight temperature scaling for coherent generation
        vocab_size_real = len(lang.id2word)

        for i in range(5):
            # Pick a random valid token to prime the network
            start_id = pad_idx
            while start_id == pad_idx or lang.id2word[start_id] in ["<pad>", "<eos>"]:
                start_id = torch.randint(0, vocab_size_real, (1,)).item()
            
            current_seq = torch.tensor([[start_id]]).to(device)
            words = [lang.id2word[start_id]]

            with torch.no_grad():
                for _ in range(25): # Limit to 25 generated tokens
                    output = model(current_seq) # [Batch, Vocab, Seq_Len]
                    logits = output[0, :, -1] / temperature
                    probs = torch.softmax(logits, dim=0)
                    next_id = torch.multinomial(probs, 1).unsqueeze(0)
                    
                    word = lang.id2word[next_id.item()]
                    if word == "<eos>":
                        break
                    
                    words.append(word)
                    current_seq = torch.cat([current_seq, next_id], dim=1)
            
            print(f"Sample {i+1}: {' '.join(words)}.")
        print("===================================\n")
        return

    # ================= TUNING MODE =================
    if args.tune:
        tuning_grid = config.get('tuning_grid', {})
        
        param_tuning_order = [
            {
                "name": "non_mono",
                # 1 was the best so we moved from 3 to 1
                "values": tuning_grid.get("non_mono", [1, 2, 3, 4, 5])
            },
        ]

        def run_single_trial(trial_cfg):
            # Local reproducibility
            set_seed(1234)
            device_local = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # Prepare datasets and loaders with trial params
            train_dataset_local = PennTreeBank(train_raw, lang)
            dev_dataset_local = PennTreeBank(dev_raw, lang)
            test_dataset_local = PennTreeBank(test_raw, lang)

            train_loader_local = DataLoader(
                train_dataset_local,
                batch_size=trial_cfg.get('batch_size', batch_size),
                collate_fn=partial(collate_fn, pad_token=pad_idx, device=device_local),
                shuffle=True
            )
            dev_loader_local = DataLoader(
                dev_dataset_local,
                batch_size=trial_cfg.get('batch_size', batch_size),
                collate_fn=partial(collate_fn, pad_token=pad_idx, device=device_local)
            )
            test_loader_local = DataLoader(
                test_dataset_local,
                batch_size=trial_cfg.get('batch_size', batch_size),
                collate_fn=partial(collate_fn, pad_token=pad_idx, device=device_local)
            )

            emb_sz = trial_cfg.get('emb_size', emb_size)
            hid_sz = trial_cfg.get('hidden_size', hid_size)
            emb_sz, hid_sz = resolve_model_sizes(emb_sz, hid_sz)
            lr_trial = trial_cfg.get('lr', lr)
            clip_trial = trial_cfg.get('clip', clip)
            n_epochs_trial = trial_cfg.get('n_epochs', n_epochs)
            patience_trial = trial_cfg.get('patience', patience)
            current_model_type = trial_cfg.get('model_type', model_type)

            # Model creation arguments mapped dynamically
            model_kwargs = {
                "emb_size": emb_sz,
                "hidden_size": hid_sz,
                "output_size": vocab_len,
                "pad_index": pad_idx,
                "emb_dropout": trial_cfg.get('emb_drop', emb_drop),
                "out_dropout": trial_cfg.get('out_drop', out_drop),
                "weight_tying": trial_cfg.get('weight_tying', weight_tying)
            }

            # Model selection handling
            if current_model_type == "LSTM":
                lstm_kwargs = {k: v for k, v in model_kwargs.items() if k != 'weight_tying'}
                model_local = LM_LSTM(**lstm_kwargs).to(device_local)
            else:
                model_local = LM_RNN(**model_kwargs).to(device_local)

            model_local.apply(init_weights)

            # Optimizer selection
            optimizer_name_trial = trial_cfg.get('optimizer', optimizer_name)
            if optimizer_name_trial in ["SGD", "NT-ASGD"]:
                optimizer_local = optim.SGD(model_local.parameters(), lr=lr_trial)
            else:
                optimizer_local = optim.SGD(model_local.parameters(), lr=lr_trial) # Fallback

            is_asgd_local = False
            non_mono_trial = trial_cfg.get('non_mono', non_mono)

            criterion_train_local = nn.CrossEntropyLoss(ignore_index=pad_idx)
            criterion_eval_local = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

            # Training loop with early stopping
            best_ppl_local = math.inf
            best_state_local = None
            current_pat = patience_trial
            losses_train_local = []
            losses_dev_local = []

            for _ in range(1, n_epochs_trial + 1):
                train_loss_local = train_loop(train_loader_local, optimizer_local, criterion_train_local, model_local, clip_trial)
                losses_train_local.append(train_loss_local)
                
                # Validation evaluation with ASGD weight swapping if active
                if optimizer_name_trial == "NT-ASGD" and is_asgd_local:
                    backup_w = swap_asgd_weights(model_local, optimizer_local)
                    ppl_dev_local, loss_dev_local = eval_loop(dev_loader_local, criterion_eval_local, model_local)
                    restore_asgd_weights(model_local, backup_w)
                else:
                    ppl_dev_local, loss_dev_local = eval_loop(dev_loader_local, criterion_eval_local, model_local)
                    
                losses_dev_local.append(loss_dev_local)
                
                # NT-ASGD Trigger Check
                if optimizer_name_trial == "NT-ASGD" and not is_asgd_local:
                    if check_nt_asgd_trigger(losses_dev_local, non_mono_trial):
                        optimizer_local = optim.ASGD(model_local.parameters(), lr=lr_trial, t0=0, lambd=0.)
                        is_asgd_local = True
                
                if ppl_dev_local < best_ppl_local:
                    best_ppl_local = ppl_dev_local
                    # Save ASGD weights if active, otherwise standard weights
                    if is_asgd_local:
                        backup_w = swap_asgd_weights(model_local, optimizer_local)
                        best_state_local = {k: v.cpu() for k, v in model_local.state_dict().items()}
                        restore_asgd_weights(model_local, backup_w)
                    else:
                        best_state_local = {k: v.cpu() for k, v in model_local.state_dict().items()}
                    current_pat = patience_trial
                else:
                    current_pat -= 1
                    if current_pat <= 0:
                        break

            # Restore best model and evaluate on test
            if current_model_type == "LSTM":
                best_model_local = LM_LSTM(**lstm_kwargs).to(device_local)
            else:
                best_model_local = LM_RNN(**model_kwargs).to(device_local)

            best_model_local.load_state_dict(best_state_local)

            final_ppl_local, _ = eval_loop(test_loader_local, criterion_eval_local, best_model_local)

            # Save experiment artifacts for this trial
            hyperparams = {k: trial_cfg.get(k, base) for k, base in [
                ("emb_size", emb_size), ("hidden_size", hid_size), ("lr", lr), 
                ("batch_size", batch_size), ("optimizer", optimizer_name), 
                ("model_type", model_type), ("patience", patience), ("clip", clip), 
                ("n_epochs", n_epochs), ("weight_tying", weight_tying), 
                ("emb_drop", emb_drop), ("out_drop", out_drop), ("non_mono", non_mono)
            ]}
            save_experiment(best_model_local, hyperparams, losses_train_local, losses_dev_local, name=trial_cfg['experiment_name'])

            extras = {"best_val_ppl": best_ppl_local, "final_test_ppl": final_ppl_local}
            return final_ppl_local, extras

        # Run sequential grid search
        search_results = sequential_grid_search(
            param_tuning_order=param_tuning_order,
            base_config=config,
            run_fn=run_single_trial,
            base_results_dir=os.path.join("bin", config.get('experiment_name', 'grid_search'))
        )
        
        # Display final results
        print("\n" + "="*70)
        print("SEQUENTIAL GRID SEARCH COMPLETE")
        print("="*70)
        print(f"Final Best Configuration:")
        for key, val in search_results['best_config'].items():
            print(f"  {key}: {val}")
        print(f"Results saved to: {search_results['results_dir']}")
        print("="*70 + "\n")
        
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
    model_kwargs = {
        "emb_size": emb_size,
        "hidden_size": hid_size,
        "output_size": vocab_len,
        "pad_index": pad_idx,
        "emb_dropout": emb_drop,
        "out_dropout": out_drop,
        "weight_tying": weight_tying
    }
    
    if model_type == "LSTM":
        lstm_kwargs = {k: v for k, v in model_kwargs.items() if k != 'weight_tying'}
        model = LM_LSTM(**lstm_kwargs).to(device)
    else:
        model = LM_RNN(**model_kwargs).to(device)
        
    model.apply(init_weights)

    if optimizer_name in ["SGD", "NT-ASGD"]:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    is_asgd = False

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
        
        # Run validation epoch with ASGD weights if active
        if optimizer_name == "NT-ASGD" and is_asgd:
            backup_w = swap_asgd_weights(model, optimizer)
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            restore_asgd_weights(model, backup_w)
        else:
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            
        losses_dev.append(loss_dev)
        
        pbar.set_description(f"Epoch {epoch} | Val PPL: {ppl_dev:.2f}{' (ASGD)' if is_asgd else ''}")
        
        # NT-ASGD Trigger Check
        if optimizer_name == "NT-ASGD" and not is_asgd:
            if check_nt_asgd_trigger(losses_dev, non_mono):
                print(f"\n[NT-ASGD] Triggered at Epoch {epoch} due to non-monotonic metric behavior.")
                # 't0=0' starts averaging immediately; 'lambd=0.' disables ASGD's internal decay factor
                optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.)
                is_asgd = True

        # Update best model weights or decrement early stopping patience
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            if is_asgd:
                backup_w = swap_asgd_weights(model, optimizer)
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                restore_asgd_weights(model, backup_w)
            else:
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            current_patience = patience  # Reset validation tracking steps
        else:
            current_patience -= 1
            if current_patience <= 0:
                print(f"\n[Early Stopping] Triggered at Epoch {epoch} due to plateauing validation loss.")
                break

    # Restore the best performing parameters from the training run
    if model_type == "LSTM":
        best_model = LM_LSTM(**lstm_kwargs).to(device)
    else:
        best_model = LM_RNN(**model_kwargs).to(device)
        
    best_model.load_state_dict(best_model_state)

    # Final evaluate run across testing dataset
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print(f"\n{'='*48}")
    print(f"Single Run Evaluation Results:")
    print(f"Best Validation Perplexity (PPL): {best_ppl:.3f}")
    print(f"Final Test Set Perplexity (PPL): {final_ppl:.3f}")
    print(f"{model_type} Test Perplexity (PPL): {final_ppl:.3f}")
    print(f"{'='*48}")

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
        "weight_tying": weight_tying,
        "emb_drop": emb_drop,
        "out_drop": out_drop,
        "non_mono": non_mono,
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