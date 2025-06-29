# For memory and inference time evaluation of different encoder architectures for motif calling.
import os
import time
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

from method_2 import (
    MotifSeqDataset,
    train_model
)

from method_1 import Encoder as Encoder_CNNTransformer

from method_3 import (
    Encoder_CNN_BiLSTM,
    Encoder_CNN_BiGRU,
    Encoder_PureBiLSTM,
    Encoder_PureBiGRU
)

#######################################
# Encoder Variants
#######################################
# (1) CNN+Transformer Encoder

# (2) CNN+BiLSTM Encoder

# (3) CNN+BiGRU Encoder

# (4) Pure BiLSTM Encoder (No CNN)

# (5) Pure BiGRU Encoder (No CNN)

    
#######################################
# Inference Time and Memory Usage Evaluation
#######################################
def evaluate_inference(model, dummy_input, n_iters=50, device="cpu"):
    # Prepare model for evaluation and move dummy input to the correct device
    model.eval()
    dummy_input = dummy_input.to(device)
    
    # Warm up the GPU or CPU by running a few forward passes without timing
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input, teacher_forcing_ratio=0.0, max_length=20)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    
    # Reset timing and memory statistics
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    # Measure inference time over n_iters runs
    with torch.no_grad():
        for _ in range(n_iters):
            start = time.perf_counter()
            _ = model(dummy_input, teacher_forcing_ratio=0.0, max_length=20)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            # Convert seconds to milliseconds
            times.append((end - start) * 1000)
    
    # Compute average inference time in milliseconds
    avg_time = np.mean(times)
    
    # Retrieve peak GPU memory usage in megabytes, if on CUDA
    peak_mem = None
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    return avg_time, peak_mem


#######################################
# Main Execution: Comparing Encoder Architectures and Evaluating Inference Speed/Memory
#######################################
if __name__ == "__main__":
    # Select the five encoder variants to compare.
    # Using CNN+Transformer, CNN+BiLSTM, CNN+BiGRU, Pure BiLSTM, and Pure BiGRU.
    encoder_variants = {
        "CNN_Transformer": Encoder_CNNTransformer,
        "CNN_BiLSTM": Encoder_CNN_BiLSTM,
        "CNN_BiGRU": Encoder_CNN_BiGRU,
        "BiLSTM": Encoder_PureBiLSTM,
        "BiGRU": Encoder_PureBiGRU
    }

    # File path to CSV file containing synthetic data.
    csv_path = os.path.join("squigulator", "longer_large_simplified_results.csv")
    # Use a subset (e.g. 50,000 rows) for quick experiments.
    sample_size = 50000

    # Hyperparameters (fixed for comparison)
    final_d_model = 256
    final_nhead = 4          # Only used by Transformer-based variants.
    final_num_layers = 2     # For Transformer, BiLSTM, and GRU based ones.
    final_dropout = 0.1
    final_hidden_size = 128
    teacher_forcing_init = 0.5
    teacher_forcing_final = 0.0
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 16
    num_epochs = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    results_summary = {}
    inference_times = {}
    memory_usages = {}

    # Load a test batch to serve as dummy input for inference measurement.
    dataset = MotifSeqDataset(csv_path, norm_method="robust_no_centre", debug=False, sample_size=sample_size)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq)
    # Grab the first batch from test_loader.
    dummy_batch, _, _ = next(iter(test_loader))
    dummy_batch = dummy_batch.to(device)

    for variant_name, encoder_class in encoder_variants.items():
        print(f"\n=== Running experiment with encoder variant: {variant_name} ===")
        
        # Instantiate encoder according to variant.
        if variant_name in ["CNN_Transformer", "CNN_BiLSTM", "CNN_BiGRU"]:
            # For CNN-based variants.
            if variant_name == "CNN_Transformer":
                encoder = encoder_class(input_channels=1, d_model=final_d_model, nhead=final_nhead,
                                          num_layers=final_num_layers, dropout=final_dropout)
            else:
                encoder = encoder_class(input_channels=1, d_model=final_d_model, num_layers=final_num_layers, dropout=final_dropout)
        elif variant_name in ["BiLSTM", "BiGRU"]:
            # Pure variants (no CNN backbone).
            encoder = encoder_class(d_model=final_d_model, num_layers=final_num_layers, dropout=final_dropout)
        else:
            raise ValueError("Unknown encoder variant")
        
        # Build the shared GRU-based decoder.
        decoder = Decoder(vocab_size=dataset.vocab_size, d_model=final_d_model,
                          hidden_size=final_hidden_size, num_layers=1, dropout=final_dropout)
        model = Seq2SeqMotifCaller(encoder, decoder)
        model.to(device)
        model.eval()

        # (Optional) Training could happen here; for inference evaluation, we use the untrained model.
        avg_time, peak_mem = evaluate_inference(model, dummy_batch, n_iters=50, device=device)
        inference_times[variant_name] = avg_time
        memory_usages[variant_name] = peak_mem if peak_mem is not None else 0.0

        # Optionally, you could evaluate accuracy here if needed.
        # token_acc, seq_acc, _, _ = evaluate_model_seq(model, test_loader, device, dataset.idx2motif, max_length=20)
        # results_summary[variant_name] = {"token_acc": token_acc, "seq_acc": seq_acc}

        print(f"[{variant_name}] Inference Time: {avg_time:.2f} ms per forward pass")
        if device.type == "cuda":
            print(f"[{variant_name}] Peak GPU Memory Usage: {peak_mem:.2f} MB")
        else:
            print(f"[{variant_name}] (CPU memory usage not measured)")

    # Plot Inference Time Comparison:
    variants = list(inference_times.keys())
    times = [inference_times[v] for v in variants]

    x = np.arange(len(variants))
    width = 0.5

    # Inference Time Comparison Graph
    plt.figure(figsize=(8, 4))  # Adjusted figure size
    plt.bar(x, times, width, color='skyblue')
    plt.ylabel('Inference Time (ms)', fontsize=10)
    plt.xlabel('Encoder Variant', fontsize=10)
    plt.title('Average Inference Time per Forward Pass', fontsize=12)
    plt.xticks(x, variants, rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()  # Adjust layout to prevent clipping
    os.makedirs("summary_graphs", exist_ok=True)
    plt.savefig(os.path.join("summary_graphs", "inference_time_comparison.png"), dpi=150)
    plt.close()

    # Memory Usage Comparison Graph (if using CUDA)
    if device.type == "cuda":
        mems = [memory_usages[v] for v in variants]
        plt.figure(figsize=(8, 4))  # Adjusted figure size
        plt.bar(x, mems, width, color='salmon')
        plt.ylabel('Peak Memory Usage (MB)', fontsize=10)
        plt.xlabel('Encoder Variant', fontsize=10)
        plt.title('Peak GPU Memory Usage during Inference', fontsize=12)
        plt.xticks(x, variants, rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join("summary_graphs", "memory_usage_comparison.png"), dpi=150)
        plt.close()

    # Print summary of inference metrics.
    print("\n=== Inference Metrics Summary ===")
    for variant in variants:
        mem_str = f"{memory_usages[variant]:.2f} MB" if device.type == "cuda" else "N/A"
        print(f"{variant}: Inference Time = {inference_times[variant]:.2f} ms, Peak Memory = {mem_str}")
