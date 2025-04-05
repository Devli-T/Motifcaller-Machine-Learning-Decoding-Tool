import os
import glob
import csv
import h5py
import re

# Define folders
fast5_folder = "longer_large_simplified_fast5_outputs"
fasta_folder = "longer_large_simplified_fasta_splits"
output_csv = "longer_large_simplified_results.csv"

# Get list of FAST5 files
fast5_files = glob.glob(os.path.join(fast5_folder, "*.fast5"))

def extract_index(filename):
    """Extract the numeric index from a filename like 'oligo_0_signal.fast5'."""
    basename = os.path.basename(filename)
    m = re.search(r'oligo_(\d+)_signal\.fast5', basename)
    if m:
        return int(m.group(1))
    return -1

# Sort the files numerically by oligo index
fast5_files = sorted(fast5_files, key=extract_index)

# Open CSV for writing
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # CSV header with binary_codes column now coming before motifs
    writer.writerow(["oligo_id", "binary_codes", "motifs", "sequence", "raw_signal"])
    
    # Process each FAST5 file in sorted order
    for fast5_path in fast5_files:
        # Extract the oligo id from the filename (assumes pattern "oligo_0_signal.fast5")
        basename = os.path.basename(fast5_path)
        oligo_id = basename.split("_signal")[0]  # e.g. "oligo_0"
        
        # Determine the corresponding FASTA file
        fasta_path = os.path.join(fasta_folder, f"{oligo_id}.fa")
        
        motifs = ""
        binary_codes = ""
        sequence = ""
        if os.path.exists(fasta_path):
            with open(fasta_path, "r") as f:
                lines = f.read().splitlines()
                if lines:
                    header = lines[0]
                    # Remove the leading ">" if present
                    if header.startswith(">"):
                        header = header[1:]
                    # Split header on "_motifs=" to get motifs and binary codes
                    parts = header.split("_motifs=")
                    if len(parts) == 2:
                        remainder = parts[1]
                        if "_binary=" in remainder:
                            motifs_part, binary_part = remainder.split("_binary=")
                            motifs = motifs_part.strip()
                            binary_codes = binary_part.strip()
                        else:
                            motifs = remainder.strip()
                    # The remaining lines form the sequence
                    sequence = "".join(lines[1:]).strip()
        else:
            print(f"Warning: {fasta_path} not found. Skipping FASTA info for {oligo_id}.")
        
        # Open the FAST5 file and extract the raw signal
        raw_signal = []
        with h5py.File(fast5_path, "r") as fast5:
            # Use a mutable container to store the dataset path
            signal_dataset_path = [None]
            
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("Raw/Signal"):
                    signal_dataset_path[0] = name

            fast5.visititems(visitor)
            
            if signal_dataset_path[0] is not None:
                raw_signal = fast5[signal_dataset_path[0]][:]
                raw_signal = raw_signal.tolist()
            else:
                print(f"Warning: No raw signal found in {fast5_path}.")

        # Convert the raw signal list to a string (semicolon-separated)
        raw_signal_str = ";".join(map(str, raw_signal))
        # Write CSV row with binary_codes column before motifs
        writer.writerow([oligo_id, binary_codes, motifs, sequence, raw_signal_str])

print(f"CSV file '{output_csv}' has been created.")
