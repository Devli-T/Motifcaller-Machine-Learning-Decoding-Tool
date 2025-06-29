# To recreate different normalisation methods graph.
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

from method_1 import (
    normalise_signal,
    collate_fn_seq,
    Encoder,
    Decoder,
    PositionalEncoding,
    Seq2SeqMotifCaller,
    greedy_decode_seq,
    evaluate_model_seq
)

# -------------------------
# Global Tokens and Settings
# -------------------------
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

# -------------------------
# Dataset Class: MotifSeqDataset (with added parameter of sample_size)
# -------------------------
class MotifSeqDataset(Dataset):
    def __init__(self, csv_path, norm_method="robust_no_centre", debug=False, sample_size=None):
        self.debug = debug
        self.norm_method = norm_method  # Store the chosen normalisation method
        self.results_df = pd.read_csv(csv_path)
        if sample_size is not None:
            self.results_df = self.results_df.head(sample_size)
        if self.debug:
            print(f"[DEBUG] Loaded {len(self.results_df)} entries from {csv_path}.")
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
        if self.debug:
            print(f"[DEBUG] Vocabulary size (including special tokens): {self.vocab_size}")
            
    def __len__(self):
        return len(self.results_df)
    
    def __getitem__(self, idx):
        row = self.results_df.iloc[idx]
        raw_signal = np.array([float(x) for x in row["raw_signal"].split(';')])
        norm_signal = normalise_signal(raw_signal, method=self.norm_method)
        signal_tensor = torch.tensor(norm_signal, dtype=torch.float).unsqueeze(0)  # (1, T)
        motifs = [m.strip() for m in row["motifs"].split(',')]
        target_indices = [SOS_TOKEN] + [self.motif2idx.get(m, PAD_TOKEN) for m in motifs] + [EOS_TOKEN]
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        return signal_tensor, target_tensor
    
# -------------------------
# Training Loop with Early Stopping and LR Scheduler
# -------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=30, lr=1e-3, weight_decay=1e-4,
                grad_clip=1.0, max_length=100, teacher_forcing_init=0.5, teacher_forcing_final=0.0,
                decay_rate=50, patience=5, save_dir="graphs"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    train_losses = []
    val_losses = []
    token_accs = []
    seq_accs = []
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
            outputs = model(signals, targets[:, :-1], teacher_forcing_ratio=teacher_forcing_ratio,
                            max_length=targets.size(1) - 1)
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
        
        # Update LR scheduler.
        scheduler.step(avg_val_loss)
        
        token_acc, seq_acc, _, _ = evaluate_model_seq(model, val_loader, device, max_length=targets.size(1) - 1)
        token_accs.append(token_acc)
        seq_accs.append(seq_acc)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        print(f"           Token Accuracy = {token_acc*100:.2f}%, Sequence Accuracy = {seq_acc*100:.2f}%")
        
        # Early stopping check.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training graphs.
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(token_accs)+1), token_accs, label="Token Accuracy")
    plt.plot(range(1, len(seq_accs)+1), seq_accs, label="Sequence Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_curves.png"))
    plt.close()

    return model, best_val_loss


# -------------------------
# Main Execution: Comparing Normalisation Methods
# -------------------------
if __name__ == "__main__":
    # List of normalisation methods to compare
    norm_methods = ["zscore", "minmax", "robust_no_centre"]
    
    # File path to the CSV file containing synthetic data
    csv_path = os.path.join("squigulator", "longer_large_simplified_results.csv")
    # Use a subset of data for quicker graph generation (e.g. first 50,000 rows)
    sample_size = 50000

    # Hyperparameters (fixed for comparison)
    final_d_model = 256
    final_nhead = 4
    final_num_encoder_layers = 2
    final_dropout = 0.1
    final_hidden_size = 128
    teacher_forcing_init = 0.5
    teacher_forcing_final = 0.0
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 16
    num_epochs = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[INFO] Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU.")

    # To store final evaluation metrics for each normalisation method
    results_summary = {}

    for norm in norm_methods:
        print(f"\n=== Running experiment with normalisation method: {norm} ===")
        # Create a save directory for graphs specific to this normalisation method
        save_dir = f"graphs_{norm}"
        os.makedirs(save_dir, exist_ok=True)

        # Load dataset with the current normalisation method
        dataset = MotifSeqDataset(csv_path, norm_method=norm, debug=True, sample_size=sample_size)
        total_samples = len(dataset)
        # Split indices into training, validation, and test sets (60/20/20 split)
        indices = list(range(total_samples))
        train_size = int(0.6 * total_samples)
        val_size = int(0.2 * total_samples)
        train_indices, rest_indices = train_test_split(indices, train_size=train_size, random_state=42)
        val_indices, test_indices = train_test_split(rest_indices, train_size=val_size, random_state=42)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_seq)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq)

        # Build the model with fixed hyperparameters
        encoder = Encoder(input_channels=1, d_model=final_d_model, nhead=final_nhead,
                          num_layers=final_num_encoder_layers, dropout=final_dropout)
        decoder = Decoder(vocab_size=dataset.vocab_size, d_model=final_d_model,
                          hidden_size=final_hidden_size, num_layers=1, dropout=final_dropout)
        model = Seq2SeqMotifCaller(encoder, decoder)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        # Train the model
        model, best_val_loss = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs,
                                           lr=learning_rate, weight_decay=weight_decay, max_length=20,
                                           teacher_forcing_init=teacher_forcing_init, teacher_forcing_final=teacher_forcing_final,
                                           save_dir=save_dir)

        # Evaluate on the test set
        token_acc, seq_acc, preds, targets = evaluate_model_seq(model, test_loader, device, dataset.idx2motif, max_length=20)
        print(f"[RESULT] Normalisation: {norm} | Test Token Accuracy: {token_acc*100:.2f}% | Test Sequence Accuracy: {seq_acc*100:.2f}%")
        results_summary[norm] = {"token_acc": token_acc, "seq_acc": seq_acc}

        # Optionally, save the trained model state for this normalisation method
        # os.makedirs(save_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(save_dir, "best_seq2seq_model.pth"))

    # Create a summary comparison plot of the final test accuracies.
    methods = list(results_summary.keys())
    token_accuracies = [results_summary[m]["token_acc"] * 100 for m in methods]
    seq_accuracies = [results_summary[m]["seq_acc"] * 100 for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, token_accuracies, width, label='Token Accuracy')
    plt.bar(x + width/2, seq_accuracies, width, label='Sequence Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Normalisation Method')
    plt.title('Test Accuracy Comparison for Different Normalisation Methods')
    plt.xticks(x, methods)
    plt.legend()
    os.makedirs("summary_graphs", exist_ok=True)
    plt.savefig(os.path.join("summary_graphs", "normalisation_comparison.png"))
    plt.close()

    print("\n=== Summary of Results ===")
    for norm, metrics in results_summary.items():
        print(f"Method: {norm} | Token Accuracy: {metrics['token_acc']*100:.2f}% | Sequence Accuracy: {metrics['seq_acc']*100:.2f}%")
