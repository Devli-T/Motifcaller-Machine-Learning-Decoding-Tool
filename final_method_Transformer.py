#CNN-Transformer-CTC
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

from method_5 import Decoder_CTC as Decoder

# -------------------------
# Global Tokens and Settings
# -------------------------
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
USE_OPTUNA = True   # Set to True to run hyperparameter tuning

# -------------------------
# Signal Normalisation Function
# -------------------------
def normalise_signal(signal, method="robust_no_center"):
    if method == "zscore":
        std = np.std(signal)
        return (signal - np.mean(signal)) / (std if std > 0 else 1.0)
    elif method == "minmax":
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / ((max_val - min_val) if (max_val - min_val) > 0 else 1.0)
    elif method == "robust_no_center":
        q1 = np.percentile(signal, 25)
        q3 = np.percentile(signal, 75)
        iqr = q3 - q1 if (q3 - q1) > 0 else 1.0
        return signal / iqr
    else:
        raise ValueError("Unknown normalization method")

# -------------------------
# Dataset Class: MotifSeqDataset
# -------------------------
class MotifSeqDataset(Dataset):
    def __init__(self, csv_path, norm_method="robust_no_center", debug=False):
        self.debug = debug
        self.results_df = pd.read_csv(csv_path)
        self.results_df = self.results_df[:50000]
        if self.debug:
            print(f"[DEBUG] Loaded {len(self.results_df)} entries from {csv_path}.", flush=True)
        # Build vocabulary from the motifs column.
        motifs_all = []
        for mstr in self.results_df["motifs"]:
            motifs = [m.strip() for m in mstr.split(',')]
            motifs_all.extend(motifs)
        unique_motifs = sorted(list(set(motifs_all)))
        self.motif2idx = {"<PAD>": PAD_TOKEN, "<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN}
        self.idx2motif = {PAD_TOKEN: "<PAD>", SOS_TOKEN: "<SOS>", EOS_TOKEN: "<EOS>"}
            
        next_idx = 3
        for motif in unique_motifs:
            self.motif2idx[motif] = next_idx
            self.idx2motif[next_idx] = motif
            next_idx += 1
        self.vocab_size = next_idx
        
        json_path = os.path.splitext(csv_path)[0] + "_motif2idx.json"
        with open(json_path, "w") as jf:
            json.dump(self.motif2idx, jf, indent=2)
            print(f"[INFO] Saved motif2idx JSON to {json_path}", flush=True)
        if self.debug:
            print(f"[DEBUG] Saved motif2idx JSON to {json_path}", flush=True)
            
        if self.debug:
            print(f"[DEBUG] Vocabulary size (with special tokens): {self.vocab_size}", flush=True)
            
    def __len__(self):
        return len(self.results_df)
    
    def __getitem__(self, idx):
        row = self.results_df.iloc[idx]
        raw_signal = np.array([float(x) for x in row["raw_signal"].split(';')])
        norm_signal = normalise_signal(raw_signal, method="robust_no_center")
        signal_tensor = torch.tensor(norm_signal, dtype=torch.float).unsqueeze(0)  # (1, T)
        motifs = [m.strip() for m in row["motifs"].split(',')]
        target_indices = [SOS_TOKEN] + [self.motif2idx.get(m, PAD_TOKEN) for m in motifs] + [EOS_TOKEN]
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        return signal_tensor, target_tensor

# -------------------------
# Collate Function for Seq2Seq
# -------------------------
def collate_fn_seq(batch):
    signals = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_signal_len = max(s.shape[1] for s in signals)
    padded_signals = [F.pad(s, (0, max_signal_len - s.shape[1])) for s in signals]
    signals_tensor = torch.stack(padded_signals)
    max_target_len = max(t.shape[0] for t in targets)
    padded_targets = [F.pad(t, (0, max_target_len - t.shape[0]), value=PAD_TOKEN) for t in targets]
    targets_tensor = torch.stack(padded_targets)
    target_lengths = [t.shape[0] for t in targets]
    return signals_tensor, targets_tensor, target_lengths

# -------------------------
# Positional Encoding Module (Fixed)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)* (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1:-1:2] = torch.cos(position * div_term[:-1])
            pe[:, -1] = 0
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (T, batch, d_model)
        pos = self.pe[:, :x.size(0), :].transpose(0, 1)  # (T, 1, d_model)
        return self.dropout(x + pos)

# -------------------------
# Encoder: CNN + Transformer Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, input_channels=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (batch, channels, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # (batch, d_model, T_reduced)
        x = x.permute(2, 0, 1)  # (T_reduced, batch, d_model)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)  # (T_reduced, batch, d_model)
        return x

# -------------------------
# Decoder: CTC Decoder
# -------------------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(Decoder, self).__init__()
        # In CTC, we simply project encoder outputs to vocabulary logits.
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.0, max_length=None):
        # encoder_outputs shape: (T_enc, batch, d_model)
        logits = self.out(encoder_outputs)  # (T_enc, batch, vocab_size)
        logits = logits.transpose(0,1)       # (batch, T_enc, vocab_size)
        return logits

# -------------------------
# Seq2Seq Model: Combines Encoder and Decoder
# -------------------------
class Seq2SeqMotifCaller(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqMotifCaller, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5, max_length=100):
        encoder_outputs = self.encoder(x)
        outputs = self.decoder(encoder_outputs, targets=targets, teacher_forcing_ratio=teacher_forcing_ratio, max_length=max_length)
        return outputs

# -------------------------
# Greedy Decoding Helper for Seq2Seq
# -------------------------
def greedy_decode_seq(outputs):
    preds = outputs.argmax(dim=2)
    decoded = []
    for seq in preds.tolist():
        tokens = []
        for token in seq:
            if token == EOS_TOKEN:
                break
            tokens.append(token)
        decoded.append(tokens)
    return decoded

# -------------------------
# Evaluation Function for Seq2Seq
# -------------------------
def evaluate_model_seq(model, dataloader, device, idx2motif, max_length=100):
    model.eval()
    all_decoded_preds = []
    all_decoded_targets = []
    seq_correct = []
    flat_preds = []
    flat_truths = []
    total_token_matches = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            signals, targets, _ = batch
            signals = signals.to(device)
            targets = targets.to(device)
            outputs = model(signals, targets=None, teacher_forcing_ratio=0.0, max_length=max_length)
            decoded_preds = greedy_decode_seq(outputs)
            decoded_targets = []
            for t in targets.tolist():
                t = t[1:]
                if EOS_TOKEN in t:
                    t = t[:t.index(EOS_TOKEN)]
                decoded_targets.append(t)
            for p, t in zip(decoded_preds, decoded_targets):
                seq_correct.append(1 if p == t else 0)
                l = min(len(p), len(t))
                total_tokens += l
                total_token_matches += sum(1 for a, b in zip(p[:l], t[:l]) if a == b)

    token_acc = total_token_matches / total_tokens if total_tokens > 0 else 0.0
    seq_acc = np.mean(seq_correct) if seq_correct else 0.0
    model.train()
    return token_acc, seq_acc, all_decoded_preds, all_decoded_targets

# -------------------------
# Training Loop with Exponential Teacher Forcing, LR Plateau Scheduler, and Early Stopping
# -------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3, weight_decay=1e-4, grad_clip=1.0, max_length=100,
                teacher_forcing_init=0.5, teacher_forcing_final=0.0, decay_rate=50, patience=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Use ReduceLROnPlateau scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    train_losses = []
    val_losses = []
    token_accs = []
    seq_accs = []
    # pr_precision = []
    # pr_recall    = []
    # pr_f1        = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        # Exponential decay for teacher forcing ratio.
        teacher_forcing_ratio = teacher_forcing_final + (teacher_forcing_init - teacher_forcing_final) * np.exp(-epoch / decay_rate)
        print(f"Epoch {epoch}: Teacher Forcing Ratio = {teacher_forcing_ratio:.3f}")
        for batch in train_loader:
            signals, targets, _ = batch
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(signals, targets[:, :-1], teacher_forcing_ratio=teacher_forcing_ratio, max_length=targets.size(1) - 1)
            min_len = min(outputs.size(1), targets.size(1) - 1)
            loss = criterion(outputs[:, :min_len, :].reshape(-1, model.decoder.out.out_features),
                             targets[:, 1:min_len+1].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                signals, targets, _ = batch
                signals = signals.to(device)
                targets = targets.to(device)
                outputs = model(signals, targets=None, teacher_forcing_ratio=0.0, max_length=targets.size(1) - 1)
                min_len = min(outputs.size(1), targets.size(1) - 1)
                loss = criterion(outputs[:, :min_len, :].reshape(-1, model.decoder.out.out_features),
                                 targets[:, 1:min_len+1].reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update LR scheduler based on val loss.
        scheduler.step(avg_val_loss)
        
        token_acc, seq_acc, all_preds, all_tgts = evaluate_model_seq(model, val_loader, device, None, max_length=targets.size(1) - 1)
        
        token_accs.append(token_acc)
        seq_accs.append(seq_acc)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print(f"           Token Accuracy = {token_acc*100:.2f}%, Sequence Accuracy = {seq_acc*100:.2f}%")
        
        with torch.no_grad():
            sample_signals, sample_targets, _ = next(iter(val_loader))
            sample_signals = sample_signals.to(device)
            outputs = model(sample_signals, targets=None, teacher_forcing_ratio=0.0, max_length=sample_targets.size(1) - 1)
            decoded = greedy_decode_seq(outputs)
            sample_target = sample_targets[0].tolist()[1:]
            if EOS_TOKEN in sample_target:
                sample_target = sample_target[:sample_target.index(EOS_TOKEN)]
            print(f"Epoch {epoch} Sample Prediction (token IDs): {decoded[0]}")
            print(f"Epoch {epoch} Ground Truth (token IDs): {sample_target}")
        
        # Early stopping check.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    os.makedirs("CLI_Tools", exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("CLI_Tools/loss_curves.png")
    plt.close()
    plt.figure()
    plt.plot(range(1, len(token_accs)+1), token_accs, label="Token Accuracy")
    plt.plot(range(1, len(seq_accs)+1), seq_accs, label="Sequence Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.savefig("CLI_Tools/accuracy_curves.png")
    plt.close()
    
    
    return model, best_val_loss

# -------------------------
# Optuna Objective Function (Specific for CNN-Transformer-CTC)
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
    encoder = Encoder(input_channels=1, d_model=d_model, nhead=nhead, num_layers=num_encoder_layers, dropout=dropout)
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
        final_nhead = best_params["nhead"]
        final_num_encoder_layers = best_params["num_encoder_layers"]
        final_dropout = best_params["dropout"]
        final_teacher_forcing_init = best_params["teacher_forcing_init"]
        final_teacher_forcing_final = best_params["teacher_forcing_final"]
        final_lr = best_params["lr"]
        final_weight_decay = best_params["weight_decay"]
        final_batch_size = best_params["batch_size"]
    else:
        final_d_model = 256
        final_nhead = 4
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
    encoder = Encoder(input_channels=1, d_model=final_d_model, nhead=final_nhead,
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
    
    os.makedirs("CLI_Tools", exist_ok=True)
    torch.save(trained_model, os.path.join("CLI_Tools", "best_seq2seq_model.pth"))
