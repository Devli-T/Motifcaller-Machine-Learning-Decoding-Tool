import argparse
import h5py
import csv
import json
import torch
import numpy as np

# Load normalisation and seq2seq classes for unpickling
from final_method_Transformer import (
    normalise_signal,
    Seq2SeqMotifCaller,
    Encoder,
    Decoder,
    PositionalEncoding,
    greedy_decode_seq
)


def extract_raw_signal(fast5_path):
    # Open the FAST5 file and search for the Raw/Signal dataset
    with h5py.File(fast5_path, "r") as fast5:
        path = None

        # Visitor function records the dataset path when it ends with "Raw/Signal"
        def visitor(name, obj):
            nonlocal path
            if isinstance(obj, h5py.Dataset) and name.endswith("Raw/Signal"):
                path = name

        fast5.visititems(visitor)

        # If no signal dataset is found, warn and return empty list
        if path is None:
            print(f"Warning: No raw signal found in {fast5_path}.")
            return []

        # Read the entire signal array and convert to Python list
        return fast5[path][:].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Seq2Seq motif caller: loads pickled model and does greedy decoding."
    )
    parser.add_argument("fast5_file", help="Input FAST5 file path.")
    parser.add_argument("output_file", help="Output CSV for motif sequence.")
    parser.add_argument(
        "--model_path", default="Final_Run+CLI_Tools/CNN_Transformer_CTC.pth",
        help="Path to torch.save pickled seq2seq model."
    )
    parser.add_argument(
        "--mapping_file", default="Final_Run+CLI_Tools/longer_large_simplified_results_motif2idx.json",
        help="JSON mapping of motif->index (with special tokens)."
    )
    args = parser.parse_args()

    # Read and normalise signal
    raw = extract_raw_signal(args.fast5_file)
    if not raw:
        print("No signal; exiting.")
        return
    norm = normalise_signal(np.array(raw), method="robust_no_center")
    signal_tensor = torch.tensor(norm, dtype=torch.float).unsqueeze(0).unsqueeze(0)

    # Load mapping
    with open(args.mapping_file, "r") as mf:
        motif2idx = json.load(mf)
    idx2motif = {int(v): k for k, v in motif2idx.items()}

    # Ensure classes in __main__ for unpickling
    import sys
    main_mod = sys.modules['__main__']
    for cls in (Encoder, Decoder, PositionalEncoding, Seq2SeqMotifCaller):
        setattr(main_mod, cls.__name__, cls)

    # Load pickled model
    model = torch.load(
        args.model_path,
        map_location=torch.device("cpu"),
        weights_only=False
    )
    model.eval()

    # Forward pass and greedy decoding
    with torch.no_grad():
        outputs = model(
            signal_tensor,
            targets=None,
            teacher_forcing_ratio=0.0,
            max_length=100
        )
        # outputs shape: (batch, seq_len, vocab)
        preds_list = greedy_decode_seq(outputs)
        if not preds_list:
            print("No predictions generated.")
            return
        preds = preds_list[0]

    # Map indices to motif strings
    predicted_motifs = [idx2motif.get(idx, "Unknown") for idx in preds]

    # Write to CSV
    with open(args.output_file, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["Position", "Predicted Motif"])
        for pos, motif in enumerate(predicted_motifs, start=1):
            writer.writerow([pos, motif])

    print(f"Predicted {len(predicted_motifs)} motifs saved to {args.output_file}")

if __name__ == "__main__":
    main()
