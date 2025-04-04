# TRIM (TCR-RNA Integrating Model)

Code for paper: Multimodal framework for the joint analysis of single-cell RNA and T cell receptor sequencing data predicts T cell response to cancer immunotherapy

arXiv link: [to-add]

## Installation

Follow the two steps illustrated below

1. create a conda environment using `environment.yaml` (all dependencies are included; whole process takes about 5 min):

```
conda env create -f environment.yml
```

2. install the current package in editable mode inside the conda environment:

```
pip install -e .
```

## Experiments
To run the models, please follow these steps:

1. Preprocess the RNA-seq and TCR-seq data (see example: `./analysis/HNSCC/data_preprocess/data_processing.py`)
2. Update the paths in ./run.sh
3. Execute the script to start the run

```
./run.sh
```

## Figures in the paper

Illustraive figures: made using powerpoint

Pointers for nonillustrative figures:

- `./analysis/HNSCC/2.0.eval_data_explore.py`: Fig.3 (a, c, d, e, f, g), Supp Fig.4
- `./analysis/HNSCC/2.1.eval_rna_pairwise_dist.py`: Fig.3b, Supp Fig.2a
- `./analysis/HNSCC/2.2.eval_gen.py`: Fig.4, Fig.5a, Supp Fig.5, Supp Fig.7, Supp Fig.8
- `./analysis/saliency`: Fig.5 (b,c,d), Supp Fig.9
- `./analysis/pan-cancer/2.1.evaluation.py`: Supp Fig.3c, Supp Fig.6
- `./analysis/pan-cancer/2.0.eval_rna_pairwise_dist.py`: Supp Fig.3b

