# Sequence-to-Sequence Motif Caller

## Overview
This project implements a sequence-to-sequence model for calling motifs from nanopore sequencing signal data. The current implementation utilises a convolutional encoder and transformer encoder combined with a GRU-based decoder with an attention mechanism to generate motif sequences from raw signal data. In upcoming future updates, more graphical data analysis will be provided.

## Synthetic Data Generation
To generate synthetic data using Squigulator, follow these steps:

```bash
cd squigulator
source myenv/bin/activate
cd Easy_FastA_Generation
python simplified_fasta_gen.py
cd ..
./simplified_process_oligos.sh
python simplified_create_csv.py
```

## Notable Results for Method 1

### Baseline Results
| Dataset Rows | Test Token Accuracy | Test Sequence Accuracy |
|--------------|---------------------|------------------------|
| 1,000        | 16.65%              | 0.00%                  |
| 10,000       | 47.04%              | 0.00%                  |
| 50,000       | 60.06%              | 1.77%                  |
| 100,000      | 75.65%              | 9.03%                  |

### Results Using Optuna for Hyperparameter Tuning
| Dataset Rows         | Test Token Accuracy | Test Sequence Accuracy |
|----------------------|---------------------|------------------------|
| 10,000 + Optuna      | 87.12%              | 26.20%                 |
| 50,000 + Optuna      | 99.78%              | 97.94%                 |
| 100,000 + Optuna     | 99.98%              | 99.76%                 |
(Max 100 Epoch Run with Early Stopping)

## Squigulator Reference
This project makes use of Squigulator to generate synthetic nanopore sequencing signal data. Special thanks to the developers for providing the simulation tool.

- Resource: Squigulator
  - Link: [Squigulator](https://github.com/hasindu2008/squigulator)
