# Sequence-to-Sequence Motif Caller

## Overview
This project implements a sequence-to-sequence model for calling motifs from nanopore sequencing signal data. The current implementation (Method 1) utilises a convolutional encoder and transformer encoder combined with a GRU-based decoder with an attention mechanism to generate motif sequences from raw signal data. Two additional methods will be added in future updates.

## Synthetic Data Generation
To generate synthetic data using Squigulator, follow these steps:

```bash
cd squigulator
cd Easy_FastA_Generation
python simplified_fasta_gen.py
cd ..
./simplified_process_oligos.sh
python simplified_create_csv.py
```

## Notable Results for Method 1

### Baseline Results
- Dataset Rows: 1,000
  - Test Token Accuracy: 16.65%
  - Test Sequence Accuracy: 0.00%
- Dataset Rows: 10,000
  - Test Token Accuracy: 47.04%
  - Test Sequence Accuracy: 0.00%
- Dataset Rows: 50,000
  - Test Token Accuracy: 60.06%
  - Test Sequence Accuracy: 1.77%
- Dataset Rows: 100,000
  - Test Token Accuracy: 75.65%
  - Test Sequence Accuracy: 9.03%

### Results Using Optuna for Hyperparameter Tuning
- Dataset Rows: 50,000 + Optuna
  - Test Token Accuracy: 99.78%
  - Test Sequence Accuracy: 97.94%
- Dataset Rows: 100,000 + Optuna
  - Test Token Accuracy: 99.98%
  - Test Sequence Accuracy: 99.76%

## Squigulator Reference
This project makes use of Squigulator to generate synthetic nanopore sequencing signal data. Special thanks to the developers for providing a robust and tunable simulation tool.

- Resource: Squigulator
  - Link: [Squigulator](https://github.com/hasindu2008/squigulator)
```