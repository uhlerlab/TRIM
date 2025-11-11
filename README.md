# TRIM (TCR-RNA Integrating Model)

Code for paper: Multimodal framework for the joint analysis of single-cell RNA and T cell receptor sequencing data predicts T cell response to cancer immunotherapy

## System requirements
The code has been developed and tested on a high-performance computing system running Ubuntu 20.04.4 LTS. The system is equipped with an AMD EPYC 7513 32-Core Processor, 2 TB of RAM, and eight NVIDIA RTX A6000 GPUs (each with 48 GB VRAM), using CUDA version 12.2 and NVIDIA driver version 535.104.12. The demo application of our pipeline can be run on significantly less powerful hardware, requiring:
- Linux-based system
- at least 20 GB of free disk space
- *one GPU*
- an internet connection 

## Installation

### Step 1: Create conda environment

Create a conda environment using `environment.yml` (all dependencies are included; whole process takes about 5 min):

```bash
conda env create -f environment.yml
conda activate trim
```

### Step 2: Install package

Install the current package in editable mode inside the conda environment:

```bash
pip install -e .
```

## Data Format Requirements

Before running TRIM, you need to prepare your data in the following format:

### Required Input Files

All data files should be saved as pickle files in your data directory:

1. **`data_rna.pkl`**: NumPy array of shape `(n_cells, n_genes)` containing normalized RNA-seq expression data (see preprocessing example)

2. **`data_labels.pkl`**: Pandas DataFrame with the following required columns:
   - `Tissue`: Binary indicator (0=blood, 1=tumor)
   - `Treatment Stage`: Binary indicator (0=pre-treatment, 1=post-treatment)
   - `Patient`: Patient ID (integer, 0-indexed)
   - `CDR3(Beta1)`: TCR CDR3 sequence index (integer index into `df_all_tcrs`)

3. **`data_labels_str.pkl`**: Pandas DataFrame with string versions of labels (same structure as `data_labels.pkl`)

4. **`df_all_tcrs.pkl`**: Pandas DataFrame with all unique TCR sequences as index (here, we use CDR3 amino acid sequences from beta chain)
   - Each row index should be a CDR3 amino acid sequence string
   - The `CDR3(Beta1)` column in `data_labels.pkl` should contain integer indices (0-indexed) that reference rows in this DataFrame

5. **`data_tcr.pkl`**: NumPy array of shape `(n_cells, dim_tcr)` with learned numeric TCR sequence embeddings for each cell, in the same order as column `CDR3(Beta1)` in `data_labels.pkl`, as produced by running `learn_tcr_embedding.py` (see below).

### Data Preprocessing Example

See `./analysis/HNSCC/data_preprocess/data_processing.py` for a complete example of how to:
- Load and normalize RNA-seq data
- Parse TCR sequences from metadata
- Create the required label DataFrames
- Format data for TRIM

## Quick Start

For users with preprocessed data, here's a minimal example:

```bash
# 1. Activate environment
conda activate trim

# 2. Learn TCR embeddings (update paths in the script first by updating data_path)
python learn_tcr_embedding.py

# 3. Train TRIM model
python trim.py \
    --data_parent_folder /path/to/your/data \
    --heldout_patient 0 \
    --device cuda:0
```

## Figures in the paper
This section contains code to reproduce the figures from our paper. To run TRIM on a new dataset, please follow the instructions in **Data Format Requirements** and **Quick Start**.

Illustrative figures: made using PowerPoint

Codes for non-illustrative figures can be found in `./analysis/`

