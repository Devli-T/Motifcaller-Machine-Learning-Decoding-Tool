#CNN-BiGRU-CTC
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
import json

from method_1 import (
    normalise_signal,
    MotifSeqDataset,
    collate_fn_seq,
    Encoder,
    PositionalEncoding,
    Seq2SeqMotifCaller,
    evaluate_model_seq,
    train_model
)

from method_3 import Encoder_CNN_BiGRU as Encoder

from method_5 import Decoder_CTC as Decoder

# -------------------------
# Global Tokens and Settings
# -------------------------
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
USE_OPTUNA = True   # Set to True to run hyperparameter tuning



# -------------------------
# Encoder: CNN + BiGRU Encoder
# -------------------------

# -------------------------
# Decoder: CTC Decoder
# -------------------------

# -------------------------
# Optuna Objective Function (Specific for CNN-BIGRU-CTC)
# -------------------------
def objective(trial):
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_int("nhead", 2, 8)
    # Ensure that d_model is divisible by nhead.
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned("d_model must be divisible by nhead")
    
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    teacher_forcing_init = trial.suggest_float("teacher_forcing_init", 0.4, 0.8)
    teacher_forcing_final = trial.suggest_float("teacher_forcing_final", 0.0, 0.2)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    # Build dataset and dataloaders.
    csv_path = os.path.join("squigulator", "longer_large_simplified_results.csv")
    dataset = MotifSeqDataset(csv_path, norm_method="robust_no_center", debug=False)
    total_samples = len(dataset)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)
    indices = list(range(total_samples))
    train_indices, rest_indices = train_test_split(indices, train_size=train_size, random_state=42)
    val_indices, _ = train_test_split(rest_indices, train_size=val_size, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq)
    
    # Build model.
    encoder = Encoder(input_channels=1, d_model=d_model, num_layers=num_encoder_layers, dropout=dropout)
    decoder = Decoder(vocab_size=dataset.vocab_size, d_model=d_model, dropout=dropout)
    model = Seq2SeqMotifCaller(encoder, decoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Train for 30 epochs during tuning.
    _, avg_val_loss = train_model(model, train_loader, val_loader, device, num_epochs=30, lr=learning_rate,
                                  weight_decay=weight_decay, max_length=20,
                                  teacher_forcing_init=teacher_forcing_init, teacher_forcing_final=teacher_forcing_final)
    return avg_val_loss

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    csv_path = os.path.join("squigulator", "longer_large_simplified_results.csv")
    dataset = MotifSeqDataset(csv_path, norm_method="robust_no_center", debug=True)
    total_samples = len(dataset)
    train_size = int(0.6 * total_samples)
    val_size = int(0.2 * total_samples)
    indices = list(range(total_samples))
    train_indices, rest_indices = train_test_split(indices, train_size=train_size, random_state=42)
    val_indices, test_indices = train_test_split(rest_indices, train_size=val_size, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_seq)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_seq)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_seq)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel.", flush=True)
    
    # Hyperparameter tuning with Optuna.
    if USE_OPTUNA:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        best_params = study.best_trial.params
        print("Best hyperparameters found:")
        print(best_params)
        final_d_model = best_params["d_model"]
        final_num_encoder_layers = best_params["num_encoder_layers"]
        final_dropout = best_params["dropout"]
        final_teacher_forcing_init = best_params["teacher_forcing_init"]
        final_teacher_forcing_final = best_params["teacher_forcing_final"]
        final_lr = best_params["lr"]
        final_weight_decay = best_params["weight_decay"]
        final_batch_size = best_params["batch_size"]
    else:
        final_d_model = 256
        final_num_encoder_layers = 2
        final_dropout = 0.1
        final_teacher_forcing_init = 0.5
        final_teacher_forcing_final = 0.0
        final_lr = 1e-3
        final_weight_decay = 1e-4
        final_batch_size = 16

    # Rebuild dataloaders with final batch size.
    train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True, collate_fn=collate_fn_seq)
    val_loader = DataLoader(val_dataset, batch_size=final_batch_size, shuffle=False, collate_fn=collate_fn_seq)
    test_loader = DataLoader(test_dataset, batch_size=final_batch_size, shuffle=False, collate_fn=collate_fn_seq)
    
    # Build final model.
    encoder = Encoder(input_channels=1, d_model=final_d_model,
                      num_layers=final_num_encoder_layers, dropout=final_dropout)
    decoder = Decoder(vocab_size=dataset.vocab_size, d_model=final_d_model,
                        dropout=final_dropout)
    model = Seq2SeqMotifCaller(encoder, decoder)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Final training for a larger number of epochs (e.g., 100 epochs).
    NUM_EPOCHS_FINAL = 100
    trained_model, _ = train_model(model, train_loader, val_loader, device, num_epochs=NUM_EPOCHS_FINAL,
                                   lr=final_lr, weight_decay=final_weight_decay, max_length=20,
                                   teacher_forcing_init=final_teacher_forcing_init, teacher_forcing_final=final_teacher_forcing_final)
 
    token_acc, seq_acc, _, _ = evaluate_model_seq(trained_model, test_loader, device, dataset.idx2motif, max_length=20)
    print(f"Test Token Accuracy: {token_acc*100:.2f}%, Test Sequence Accuracy: {seq_acc*100:.2f}%")
    
    os.makedirs("CCLI_Tools", exist_ok=True)
    torch.save(trained_model, os.path.join("CCLI_Tools", "best_seq2seq_model.pth"))
