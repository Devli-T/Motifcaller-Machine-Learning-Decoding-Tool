#!/bin/bash
# Create directories to store individual FASTA files and squigulator outputs
mkdir -p longer_large_simplified_fasta_splits
mkdir -p longer_large_simplified_blow5_outputs

# Path to the production_oligos.fa file (found in the "Synthetic FastA Generation" folder)
INPUT_FASTA="Easy_FastA_Generation/longer_large_production_oligos.fa"

# Split production_oligos.fa into separate files (one per oligo)
# This AWK command assumes each header begins with ">oligo_<number>"
awk '/^>/{ 
    if (f) close(f); 
    match($0, /^>(oligo_[0-9]+)/, arr); 
    f = "longer_large_simplified_fasta_splits/" arr[1] ".fa" 
} 
{ print >> f }' "$INPUT_FASTA"

# Loop over each split file and run squigulator
for fasta in longer_large_simplified_fasta_splits/*.fa; do
    # Extract the base name (e.g. oligo_0)
    base=$(basename "$fasta" .fa)
    # Define the output file name for the squigulator result
    output="longer_large_simplified_blow5_outputs/${base}_signal.blow5"
    
    echo "Processing $fasta -> $output"
    
    # Run squigulator with the current FASTA file
    ./squigulator "$fasta" -x dna-r9-prom -o "$output" -n 1
done

# Convert all blow5 files to FAST5 using slow5tools
echo "Converting blow5 files to FAST5 using slow5tools..."
slow5tools/./slow5tools s2f longer_large_simplified_blow5_outputs -d longer_large_simplified_fast5_outputs
