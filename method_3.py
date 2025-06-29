# For Encoder Graphs
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
    Decoder,
    PositionalEncoding,
    Seq2SeqMotifCaller,
    greedy_decode_seq,
    evaluate_model_seq
)

from method_1 import Encoder as Encoder_CNNTransformer

from method_2 import (
    MotifSeqDataset,
    train_model
)


# -------------------------
# Encoder Variants
# -------------------------
# (1) CNN+Transformer Encoder (CNN backbone + Transformer layers)


# (2) CNN+BiLSTM Encoder (CNN backbone + BiLSTM)
class Encoder_CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=1, d_model=256, num_layers=2, dropout=0.1):
        super(Encoder_CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bi_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model//2,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
    def forward(self, x):
        # x: (batch, 1, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # (batch, d_model, T_reduced)
        x = x.permute(0, 2, 1)  # (batch, T_reduced, d_model)
        x, _ = self.bi_lstm(x)  # (batch, T_reduced, d_model)
        x = self.fc(x)
        x = x.transpose(0, 1)  # (T_reduced, batch, d_model)
        x = self.pos_enc(x)
        return x

# (3) CNN+Vanilla RNN Encoder (CNN backbone + RNN)
class Encoder_CNN_RNN(nn.Module):
    def __init__(self, input_channels=1, d_model=256, num_layers=2, dropout=0.1):
        super(Encoder_CNN_RNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.rnn = nn.RNN(input_size=d_model, hidden_size=d_model,
                          num_layers=num_layers, batch_first=True,
                          nonlinearity='tanh', dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
    def forward(self, x):
        # x: (batch, 1, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # (batch, d_model, T_reduced)
        x = x.permute(0, 2, 1)  # (batch, T_reduced, d_model)
        x, _ = self.rnn(x)      # (batch, T_reduced, d_model)
        x = self.fc(x)
        x = x.transpose(0, 1)   # (T_reduced, batch, d_model)
        x = self.pos_enc(x)
        return x

# (4) CNN+Feedforward Encoder (CNN+FF)
class Encoder_CNN_FF(nn.Module):
    def __init__(self, input_channels=1, d_model=256, dropout=0.1):
        super(Encoder_CNN_FF, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
    def forward(self, x):
        # x: (batch, 1, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)   # (batch, d_model, T_reduced)
        x = x.permute(0, 2, 1)  # (batch, T_reduced, d_model)
        x = self.ff(x)         # (batch, T_reduced, d_model)
        x = x.transpose(0, 1)   # (T_reduced, batch, d_model)
        x = self.pos_enc(x)
        return x

# (5) Pure Transformer Encoder (No CNN)
class Encoder_Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super(Encoder_Transformer, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (batch, 1, T) -> (batch, T, 1)
        x = x.squeeze(1).unsqueeze(2)
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        return x

# (6) Raw Input Encoder (Minimal Processing)
class Encoder_Raw(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=5000):
        super(Encoder_Raw, self).__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        
    def forward(self, x):
        # x: (batch, 1, T) -> (batch, T, 1)
        x = x.squeeze(1).unsqueeze(2)
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_enc(x)
        return x

# (7) Pure BiLSTM Encoder (No CNN)
class Encoder_PureBiLSTM(nn.Module):
    def __init__(self, d_model=256, num_layers=2, dropout=0.1):
        super(Encoder_PureBiLSTM, self).__init__()
        self.bi_lstm = nn.LSTM(input_size=1, hidden_size=d_model//2, num_layers=num_layers,
                               batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
    def forward(self, x):
        # x: (batch, 1, T) -> (batch, T, 1)
        x = x.squeeze(1).unsqueeze(-1)
        output, _ = self.bi_lstm(x)
        x = self.fc(output)
        x = x.transpose(0, 1)
        x = self.pos_enc(x)
        return x

# (8) CNN+BiGRU Encoder (CNN backbone + bidirectional GRU)
class Encoder_CNN_BiGRU(nn.Module):
    def __init__(self, input_channels=1, d_model=256, num_layers=2, dropout=0.1):
        super(Encoder_CNN_BiGRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bi_gru = nn.GRU(input_size=d_model, hidden_size=d_model//2,
                             num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
    def forward(self, x):
        # x: (batch, 1, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # (batch, d_model, T_reduced)
        x = x.permute(0, 2, 1)  # (batch, T_reduced, d_model)
        x, _ = self.bi_gru(x)   # (batch, T_reduced, d_model)
        x = self.fc(x)
        x = x.transpose(0, 1)   # (T_reduced, batch, d_model)
        x = self.pos_enc(x)
        return x

# (9) Pure BiGRU Encoder (No CNN)
class Encoder_PureBiGRU(nn.Module):
    def __init__(self, d_model=256, num_layers=2, dropout=0.1):
        super(Encoder_PureBiGRU, self).__init__()
        self.bi_gru = nn.GRU(input_size=1, hidden_size=d_model//2, num_layers=num_layers,
                             batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
    def forward(self, x):
        # x: (batch, 1, T) -> (batch, T, 1)
        x = x.squeeze(1).unsqueeze(-1)
        x, _ = self.bi_gru(x)
        x = self.fc(x)
        x = x.transpose(0, 1)   # (T, batch, d_model)
        x = self.pos_enc(x)
        return x


# -------------------------
# Main Execution: Comparing Encoder Architectures
# -------------------------
if __name__ == "__main__":
    # Define all encoder variants to compare.
    # Four CNN-based variants plus pure approaches.
    encoder_variants = {
        "CNN_Transformer": Encoder_CNNTransformer,
        "CNN_BiLSTM": Encoder_CNN_BiLSTM,
        "CNN_RNN": Encoder_CNN_RNN,
        "CNN_FF": Encoder_CNN_FF,
        "Transformer": Encoder_Transformer,
        "Raw": Encoder_Raw,
        "BiLSTM": Encoder_PureBiLSTM,
        "CNN_BiGRU": Encoder_CNN_BiGRU,
        "BiGRU": Encoder_PureBiGRU
    }

    # File path to the CSV file.
    csv_path = os.path.join("squigulator", "longer_large_simplified_results.csv")
    # Use a subset (e.g. 50,000 rows) for quicker experiments.
    sample_size = 50000

    # Hyperparameters (fixed for comparison)
    final_d_model = 256
    final_nhead = 4          # Only used for Transformer-based variants.
    final_num_layers = 2     # For Transformer, BiLSTM, and RNN.
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

    results_summary = {}

    for variant_name, encoder_class in encoder_variants.items():
        print(f"\n=== Running experiment with encoder variant: {variant_name} ===")
        save_dir = f"graphs_{variant_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Load dataset (using robust normalisation).
        dataset = MotifSeqDataset(csv_path, norm_method="robust_no_centre", debug=True, sample_size=sample_size)
        total_samples = len(dataset)
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

        # Instantiate the chosen encoder with the appropriate arguments.
        if variant_name in ["CNN_Transformer", "CNN_BiLSTM", "CNN_RNN", "CNN_FF", "CNN_BiGRU"]:
            # For CNN-based variants, pass input_channels=1 and d_model.
            if variant_name == "CNN_Transformer":
                encoder = encoder_class(input_channels=1, d_model=final_d_model, nhead=final_nhead,
                                          num_layers=final_num_layers, dropout=final_dropout)
            elif variant_name == "CNN_FF":
                encoder = encoder_class(input_channels=1, d_model=final_d_model, dropout=final_dropout)
            else:
                encoder = encoder_class(input_channels=1, d_model=final_d_model, num_layers=final_num_layers, dropout=final_dropout)
        elif variant_name in ["Transformer", "Raw"]:
            # Pure Transformer and Raw take d_model and other parameters.
            if variant_name == "Transformer":
                encoder = encoder_class(d_model=final_d_model, nhead=final_nhead,
                                        num_layers=final_num_layers, dropout=final_dropout)
            else:
                encoder = encoder_class(d_model=final_d_model, dropout=final_dropout)
        elif variant_name in ["BiLSTM", "BiGRU"]:
            # Pure BiLSTM and pure BiGRU (no CNN backbone).
            encoder = encoder_class(d_model=final_d_model, num_layers=final_num_layers, dropout=final_dropout)
        else:
            raise ValueError("Unknown encoder variant")

        # Build the shared GRU-based decoder.
        decoder = Decoder(vocab_size=dataset.vocab_size, d_model=final_d_model,
                          hidden_size=final_hidden_size, num_layers=1, dropout=final_dropout)
        model = Seq2SeqMotifCaller(encoder, decoder)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        # Train the model.
        model, best_val_loss = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs,
                                           lr=learning_rate, weight_decay=weight_decay, max_length=20,
                                           teacher_forcing_init=teacher_forcing_init, teacher_forcing_final=teacher_forcing_final,
                                           save_dir=save_dir)

        # Evaluate on the test set.
        token_acc, seq_acc, preds, targets = evaluate_model_seq(model, test_loader, device, dataset.idx2motif, max_length=20)
        print(f"[RESULT] Encoder Variant: {variant_name} | Test Token Accuracy: {token_acc*100:.2f}% | Test Sequence Accuracy: {seq_acc*100:.2f}%")
        results_summary[variant_name] = {"token_acc": token_acc, "seq_acc": seq_acc}

        # Optionally, save the trained model state.
        # torch.save(model.state_dict(), os.path.join(save_dir, "best_seq2seq_model.pth"))

    # Create a summary comparison plot.
    # Reorder the methods: group CNN-based options first, then the pure approaches.
    ordered_methods = [
        "CNN_Transformer",
        "CNN_BiLSTM",
        "CNN_RNN",
        "CNN_FF",
        "CNN_BiGRU",
        "Transformer",
        "Raw",
        "BiLSTM",
        "BiGRU"
    ]

    # Extract accuracies in the desired order.
    token_accuracies = [results_summary[m]["token_acc"] * 100 for m in ordered_methods]
    seq_accuracies = [results_summary[m]["seq_acc"] * 100 for m in ordered_methods]

    # Create a comparison bar chart with improved size and font settings.
    plt.figure(figsize=(12, 8))
    x = np.arange(len(ordered_methods))
    width = 0.4

    plt.bar(x - width/2, token_accuracies, width, label='Token Accuracy')
    plt.bar(x + width/2, seq_accuracies, width, label='Sequence Accuracy')

    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xlabel('Encoder Variant', fontsize=14)
    plt.title('Test Accuracy Comparison for Encoder Variants', fontsize=16)
    plt.xticks(x, ordered_methods, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()

    os.makedirs("summary_graphs", exist_ok=True)
    plt.savefig(os.path.join("summary_graphs", "encoder_comparison.png"))
    plt.close()
