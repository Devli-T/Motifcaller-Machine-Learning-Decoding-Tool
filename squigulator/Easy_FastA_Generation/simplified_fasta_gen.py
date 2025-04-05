import random
from Bio.SeqUtils import gc_fraction

# =============================================================================
# BIOCHEMICAL CONSTRAINTS & MOTIF GENERATION
# =============================================================================
# Define biochemical constraints for motif generation.
BIOCHEM_CONSTRAINTS = {
    'gc_range': (40, 60),           # Acceptable GC percentage range (in %)
    'max_homopolymer': 4,           # Maximum allowed consecutive identical nucleotides
    'forbidden_sequences': ['GAATTC', 'GGATCC', 'AAGCTT']  # Sequences that must not be present
}

# Generate a biochemically valid DNA motif of given length.
# The motif must not be in the 'existing' set and must satisfy the constraints.
def generate_valid_motif(length: int, existing: set) -> str:
    while True:
        # Randomly generate a DNA sequence of the specified length.
        motif = ''.join(random.choices('ACGT', k=length))
        # Check that the motif is not already generated, has valid GC content,
        # does not contain a homopolymer longer than allowed, and does not include any forbidden sequences.
        if (motif not in existing and
            BIOCHEM_CONSTRAINTS['gc_range'][0] <= gc_fraction(motif) * 100 <= BIOCHEM_CONSTRAINTS['gc_range'][1] and
            not any(h * BIOCHEM_CONSTRAINTS['max_homopolymer'] in motif for h in 'ACGT') and
            not any(fs in motif for fs in BIOCHEM_CONSTRAINTS['forbidden_sequences'])):
            return motif

# Generate 8 payload motifs and store them in a list.
payload_motifs = []
existing = set()
for _ in range(8):
    motif = generate_valid_motif(10, existing)
    payload_motifs.append(motif)
    existing.add(motif)
    
# =============================================================================
# BINARY MAPPING FOR DNA STORAGE
# =============================================================================
# Map each payload motif to a unique 3-bit binary code.
motif_to_binary = {motif: format(i, '03b') for i, motif in enumerate(payload_motifs)}

# =============================================================================
# SPACER GENERATION
# =============================================================================
# Generate a spacer sequence and split it into two equal parts (front and back).
spacer = generate_valid_motif(10, existing)
spacer_front = spacer[:5]
spacer_back = spacer[5:]

# =============================================================================
# FASTA AND METADATA GENERATION FUNCTIONS
# =============================================================================
# Create a composite oligo sequence by inserting each selected motif between spacer parts.
def create_composite_sequence(selected_motifs: list) -> str:
    return ''.join([f"{spacer_front}{motif}{spacer_back}" for motif in selected_motifs])

# Write a mapping file that records the binary code corresponding to each payload motif.
def generate_payload_mapping_file(mapping_file: str, motif_to_binary: dict):
    with open(mapping_file, 'w') as f:
        f.write("binary_code,motif\n")
        for motif, binary in motif_to_binary.items():
            f.write(f"{binary},{motif}\n")

# Generate a production FASTA file and an accompanying metadata file.
# The metadata includes oligo ID, spacer sequence, selected motifs, and their binary codes.
def generate_production_fasta_and_metadata(fasta_file: str, metadata_file: str, num_oligos: int):
    with open(fasta_file, 'w') as f_fasta, open(metadata_file, 'w') as f_meta:
        # Write header for the metadata file.
        f_meta.write("oligo_id,spacer,selected_motifs,binary_codes\n")
        for i in range(num_oligos):
            # Randomly select motifs (with possible repeats) from payload_motifs.
            selected = random.choices(payload_motifs, k=10)
            seq = create_composite_sequence(selected)
            # Retrieve the binary codes for the selected motifs.
            binary_codes = [motif_to_binary[m] for m in selected]
            
            # Write FASTA entry with header containing motifs and binary codes.
            f_fasta.write(f">oligo_{i+1}_motifs={','.join(selected)}_binary={','.join(binary_codes)}\n{seq}\n")
            
            # Write metadata: oligo ID, spacer sequence, selected motifs and binary codes separated by semicolons.
            f_meta.write(f"oligo_{i+1},{spacer},{';'.join(selected)},{';'.join(binary_codes)}\n")

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Generate a payload mapping file (binary code to motif).
    generate_payload_mapping_file("longer_large_payload_mapping.csv", motif_to_binary)
    
    # Generate production dataset (e.g. 100,000 oligos) along with associated metadata.
    generate_production_fasta_and_metadata("longer_large_production_oligos.fa",
                                           "longer_large_production_oligos_metadata.csv",
                                           num_oligos=100000)
