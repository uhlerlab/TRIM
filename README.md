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

```
./run.sh
```

Source code folder: `./trim/`

## Figures in the paper

Illustraive figures: made using powerpoint

Pointers for nonillustrative figures:

- `./analysis/single_gene_perturbation`: Fig. 2, Supplementary Fig. 2-4
