# Motifcaller: A Machine Learning-Driven Decoding Tool for DNA Storage Data

## Overview
Motifcaller provides a complete framework for direct squiggle-to-motif decoding, organised into modular experiments and deployment tools. In the Normalisation_Methods folder you will find the code and results for comparing z-score, min-max and robust_no_centre preprocessing approaches (Section 8.1.1). The Varying_Dataset_Size directory contains scripts and logs showing how our CNN–Transformer–GRU (Attention) baseline scales from 1 000 to 100 000 reads (Section 8.1.2). In Encoder_Methods we implement and benchmark nine different encoder backbones against a fixed GRU-attention decoder (Section 8.1.3), while Decoder_Methods evaluates eleven decoder schemes paired with the top encoders (Section 8.1.4). Finally, Final_Run+CLI_Tools captures our fully tuned training runs and models of CNN-Transformer-CTC and CNN-BiGRU-CTC (Section 8.1.5).

## Model Scripts
Each of the Python files implements a specific experiment or final model. To run any of them, simply invoke `python [File_Name]` from the repository root. The `method_1.py` script implements the convolutional encoder with transformer layers and a GRU decoder using attention. `method_2.py` recreates the normalisation comparison graphs. `method_3.py` generates the encoder performance graphs. `method_4.py` evaluates memory usage and inference time for different encoder architectures. `method_5.py` produces the decoder options comparison graph. `final_method_BiGRU.py` contains the fully tuned CNN–BiGRU–CTC implementation, and `final_method_Transformer.py` implements the CNN–Transformer–CTC variant along with the associated CLI tools.

## Activate Environment
Before running any experiments or tools, navigate to the squigulator directory and activate the Python virtual environment. From the repository root, execute:

```bash
cd squigulator
source myenv/bin/activate
```

If you encounter missing dependencies or errors, install the required libraries listed in `requirements.txt`.

## Synthetic Data Generation
To produce the synthetic squiggle signals used throughout our experiments (as described in Chapter 6.4), first generate FASTA files, then simulate raw signals and assemble them into a CSV. From the squigulator directory, run:

```bash
cd Easy_FastA_Generation
python simplified_fasta_gen.py
cd ..
./simplified_process_oligos.sh
python simplified_create_csv.py
```
This sequence of scripts will create the FASTA inputs, invoke Squigulator to simulate Fast5 files, and extract both signal traces and motif metadata into `longer_large_simplified_results.csv`.

## Command-Line Interface (CLI)
The `predict_motifs.py` script in the Final_Run+CLI_Tools folder wraps model loading, signal normalisation and greedy decoding into a single command. By default it uses the CNN–Transformer–CTC checkpoint and the accompanying motif-to-index mapping in `CLI_Tools/longer_large_simplified_results_motif2idx.json`. To test on an example Fast5 file and compare against a ground truth list of motifs, run: 
python predict_motifs.py <fast5_file> <output_csv> [--model_path MODEL] [--mapping_file MAP]

For example:

```bash
python predict_motifs.py oligo_5_signal.fast5 oligo_5_prediction.csv
```

The output CSV will list each predicted motif in order. You can swap in any trained model checkpoint by specifying `--model_path` and `--mapping_file` as needed, provided the mapping file matches the motif vocabulary used during training.

Other working samples for the CNN-Transformer-CTC model can be found in More_Oligo_FASTA+FAST5.

## Squigulator Reference
This project makes use of Squigulator to generate synthetic nanopore sequencing signal data. Special thanks to the developers for providing the simulation tool.

- Resource: Squigulator
  - Link: [Squigulator](https://github.com/hasindu2008/squigulator)
