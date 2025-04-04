import numpy as np
import pandas as pd
import scipy.io
import sklearn.decomposition
import sklearn.svm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import sys
import os
import time
import sklearn.cluster
import logomaker as lm
import scanpy
import math
import glob
import json
import argparse
import statsmodels.api as sm
import torch
import pickle
from torch import nn
import torch.nn.functional as F
import scipy.stats as stats
import scanpy as sc
from anndata import AnnData
import matplotlib.patches as mpatches
import gseapy as gp
from gseapy import barplot, dotplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.stats import linregress
from saliency_helpers import investigate_corner_genes
fig = plt.figure()
old_fig_size = fig.get_size_inches()

# define the type of DE analysis
de_type = 'include_emergent_disappear' # include_emergent_disappear, no_emergent_include_disappear, no_emergent_no_disappear
# note: 
# include_emergent_disappear: include emergent-expanded TCRs (impacted post-trt DE) and disappeared non-expanded TCRs (impacted pre-trt DE)
# no_emergent_include_disappear: exclude emergent-expanded TCRs (impacted post-trt DE) but include disappeared non-expanded TCRs (impacted pre-trt DE)
# no_emergent_no_disappear: exclude emergent-expanded TCRs (impacted post-trt DE) and disappeared non-expanded TCRs (impacted pre-trt DE)

###############
############### START data loading
############### 
print('Starting to load data...')
t = time.time()

# Load log-fold changes and saliency values
cellType = 'CD8'

# Define the parent path and columns of interest
parent_path = f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/data/'
lfc_columns = ['lfc_post', 'lfc_pre', 'lfc_clone_pre_10', 'lfc_clone_pre_1']
lfc_columns_seurat = lfc_columns #+ ['lfc_expanded', 'lfc_nonexpanded']
lfc_columns_pval_sc = [f'{col}_pvals_adj' for col in lfc_columns]

# Manual calculations --------------------------------------------------------------------
diffexp_df_manual_raw = pd.read_csv(f'{parent_path}diffexp_df_manual.csv', index_col=0)
diffexp_df_manual = diffexp_df_manual_raw[lfc_columns+['salient_genes']]
# add '_manual' to the column names
diffexp_df_manual.columns = [f'{col}_manual' for col in diffexp_df_manual.columns]

# Scanpy calculations --------------------------------------------------------------------
diffexp_df_sc_raw = pd.read_csv(f'{parent_path}diffexp_df_sc.csv', index_col=0)
diffexp_df_sc = diffexp_df_sc_raw[lfc_columns + lfc_columns_pval_sc + ['salient_genes']]
# add '_sc' to the column names
diffexp_df_sc.columns = [f'{col}_sc' for col in diffexp_df_sc.columns]
assert diffexp_df_sc.shape[0] == diffexp_df_manual.shape[0]

# Seurat calculations --------------------------------------------------------------------
diffexp_df_seurat = pd.DataFrame()

for lfc_col in lfc_columns_seurat:
    seurat_df_temp = pd.read_csv(f'{parent_path}seurat/{lfc_col}_markers.csv', index_col=0)
    selected_cols = ['avg_log2FC', 'p_val_adj']
    seurat_df_temp = seurat_df_temp[selected_cols]
    # rename avg_log2FC to lfc_col and p_val_adj to lfc_col_pvals_adj
    seurat_df_temp.columns = [f'{lfc_col}_seurat', f'{lfc_col}_pvals_adj_seurat']
    if diffexp_df_seurat.empty:
        diffexp_df_seurat = seurat_df_temp
    else:
        assert diffexp_df_seurat.shape[0] == seurat_df_temp.shape[0], 'The number of genes should be the same'
        diffexp_df_seurat = pd.merge(diffexp_df_seurat, seurat_df_temp, left_index=True, right_index=True, how='outer')


diffexp_df_1 = pd.merge(diffexp_df_manual, diffexp_df_seurat, left_index=True, right_index=True, how='outer')
diffexp_df_all = pd.merge(diffexp_df_1, diffexp_df_sc, left_index=True, right_index=True, how='outer')
diffexp_df_all.to_csv(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/data/diffexp_df_all.csv')

### Step 0: Quick check that different methods generate similar results --------------------------------------------------------------------

# plot a scatter plot between lfc_post_manual and lfc_post_sc
manual_lfc_list = ['lfc_post_manual', 'lfc_pre_manual', 'lfc_clone_pre_10_manual', 'lfc_clone_pre_1_manual']
seurat_lfc_list = ['lfc_post_seurat', 'lfc_pre_seurat', 'lfc_clone_pre_10_seurat', 'lfc_clone_pre_1_seurat']
sc_lfc_list = ['lfc_post_sc', 'lfc_pre_sc', 'lfc_clone_pre_10_sc', 'lfc_clone_pre_1_sc']

seurat_pval_list = ['lfc_post_pvals_adj_seurat', 'lfc_pre_pvals_adj_seurat', 'lfc_clone_pre_10_pvals_adj_seurat', 'lfc_clone_pre_1_pvals_adj_seurat']
sc_pval_list = ['lfc_post_pvals_adj_sc', 'lfc_pre_pvals_adj_sc', 'lfc_clone_pre_10_pvals_adj_sc', 'lfc_clone_pre_1_pvals_adj_sc']

# Define lists for comparisons
comparison_pairs = [
    (manual_lfc_list, seurat_lfc_list),
    (manual_lfc_list, sc_lfc_list),
    # (seurat_lfc_list, sc_lfc_list)
]

# Plot scatter plots for each pair of lists
for x_list, y_list in comparison_pairs:
    for x_col, y_col in zip(x_list, y_list):
        plt.figure(figsize=(8, 6))
        plt.scatter(diffexp_df_all[x_col], diffexp_df_all[y_col], alpha=0.7)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/diff_methods_checks/{x_col}_vs_{y_col}.png')
        plt.close()

for x_col, y_col in zip(seurat_pval_list, sc_pval_list):
    plt.figure(figsize=(8, 6))
    diffexp_df_all_temp = diffexp_df_all.dropna(subset=[x_col, y_col])
    # susbet to x_col < 0.05 or y_col < 0.05
    diffexp_df_all_temp = diffexp_df_all_temp[(diffexp_df_all_temp[x_col] < 0.001) | (diffexp_df_all_temp[y_col] < 0.001)]
    print(diffexp_df_all_temp.shape)
    plt.scatter(diffexp_df_all_temp[x_col], diffexp_df_all_temp[y_col], alpha=0.7)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/diff_methods_checks/{x_col}_vs_{y_col}.png')
    plt.close()


### Step 0: Clean up the data --------------------------------------------------------------------
assert (diffexp_df_all['salient_genes_manual'] == diffexp_df_all['salient_genes_sc']). all(), 'The salient genes should be the same'
diffexp_df = diffexp_df_all.copy()
seurat_lfc_list #+= ['lfc_expanded_seurat', 'lfc_nonexpanded_seurat']
seurat_pval_list #+= ['lfc_expanded_pvals_adj_seurat', 'lfc_nonexpanded_pvals_adj_seurat']
diffexp_df = diffexp_df[seurat_lfc_list + seurat_pval_list + ['salient_genes_manual']]
# remove _seurat from the column names
diffexp_df.columns = [col.replace('_seurat', '') for col in diffexp_df.columns]
# remove _manual from the column names
diffexp_df.columns = [col.replace('_manual', '') for col in diffexp_df.columns]
assert (diffexp_df['salient_genes'] == diffexp_df_all['salient_genes_manual']).all(), 'The salient genes should the same'
assert (diffexp_df['lfc_post'] == diffexp_df_all['lfc_post_seurat']).all(), 'The log-fold changes should be the same'
assert (diffexp_df['lfc_pre'] == diffexp_df_all['lfc_pre_seurat']).all(), 'The log-fold changes should be the same'
assert (diffexp_df['lfc_clone_pre_10'] == diffexp_df_all['lfc_clone_pre_10_seurat']).all(), 'The log-fold changes should be the same'
assert (diffexp_df['lfc_clone_pre_1'] == diffexp_df_all['lfc_clone_pre_1_seurat']).all(), 'The log-fold changes should be the same'
# assert (diffexp_df['lfc_expanded'] == diffexp_df_all['lfc_expanded_seurat']).all(), 'The log-fold changes should be the same'
# assert (diffexp_df['lfc_nonexpanded'] == diffexp_df_all['lfc_nonexpanded_seurat']).all(), 'The log-fold changes should be the same'

diffexp_df['salient_genes_rank'] = diffexp_df['salient_genes'].rank(ascending=False)
diffexp_df['lfc_post_rank'] = diffexp_df['lfc_post'].rank(ascending=False)
diffexp_df['lfc_pre_rank'] = diffexp_df['lfc_pre'].rank(ascending=False)
diffexp_df['lfc_clone_pre_10_rank'] = diffexp_df['lfc_clone_pre_10'].rank(ascending=False)
diffexp_df['lfc_clone_pre_1_rank'] = diffexp_df['lfc_clone_pre_1'].rank(ascending=False)
# diffexp_df['lfc_expanded_rank'] = diffexp_df['lfc_expanded'].rank(ascending=False)
# diffexp_df['lfc_nonexpanded_rank'] = diffexp_df['lfc_nonexpanded'].rank(ascending=False)
diffexp_df.to_csv(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/data/diffexp_df.csv')

### Step 0.1: Quick check that the orders are calculated correctly --------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(diffexp_df['salient_genes'], diffexp_df['salient_genes_rank'], alpha=0.7)
plt.xlabel('Saliency Value')
plt.ylabel('Saliency Rank')
plt.title('Saliency Values vs. Saliency Ranks')
plt.grid(True)
plt.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/raw_vs_ranks/saliency_vs_saliency_ranks.png')

plt.figure(figsize=(8, 6))
plt.scatter(diffexp_df['lfc_post'], diffexp_df['lfc_post_rank'], alpha=0.7)
plt.xlabel('Log Fold Change')
plt.ylabel('Log Fold Change Rank')
plt.title('Log Fold Change (Expanded vs. non-expanded post-Treatment)')
plt.grid(True)
plt.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/raw_vs_ranks/lfc_post_vs_lfc_post_ranks.png')

plt.figure(figsize=(8, 6))
plt.scatter(diffexp_df['lfc_pre'], diffexp_df['lfc_pre_rank'], alpha=0.7)
plt.xlabel('Log Fold Change')
plt.ylabel('Log Fold Change Rank')
plt.title('Log Fold Change (Expanding vs. non-expanding)')
plt.grid(True)
plt.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/raw_vs_ranks/lfc_pre_vs_lfc_pre_ranks.png')

plt.figure(figsize=(8, 6))
plt.scatter(diffexp_df['lfc_clone_pre_10'], diffexp_df['lfc_clone_pre_10_rank'], alpha=0.7)
plt.xlabel('Log Fold Change')
plt.ylabel('Log Fold Change Rank')
plt.title('Log Fold Change (big clone vs small clone pre-treatment, threshold = 10)')
plt.grid(True)
plt.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/raw_vs_ranks/lfc_clone_pre_10.png')

plt.figure(figsize=(8, 6))
plt.scatter(diffexp_df['lfc_clone_pre_1'], diffexp_df['lfc_clone_pre_1_rank'], alpha=0.7)
plt.xlabel('Log Fold Change')
plt.ylabel('Log Fold Change Rank')
plt.title('Log Fold Change (big clone vs small clone pre-treatment, threshold = 1)')
plt.grid(True)
plt.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/raw_vs_ranks/lfc_clone_pre_1.png')


### Step 1: Plot pair-wise explorations -------------------------------------------------------------------- 

## (1) lfc_post vs. salient_genes
fig.set_size_inches([3, 5])
fig.clf()
ax1 = fig.subplots(1, 1)
x = diffexp_df['salient_genes']
y = diffexp_df['lfc_post']
ax1.scatter(x, y, s=3, alpha=0.8)
#Pearson correlation
corr, p_value = pearsonr(x, y)
formatted_pval = f"10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}" if f'{p_value:.1e}'.startswith('1e') else f"{float(f'{p_value:.1e}'.split('e')[0])} \\times 10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}"
corr_text = f"$r = {corr:.2f}, p = {formatted_pval}$"
# ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, fontsize=9,
#          verticalalignment='top')

# Fit line using linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept
ax1.plot(line_x, line_y, color='red', linestyle='-', linewidth=1, label='Fitted Line')
ax1.set_xticks(np.linspace(-0.0002, 0.0002, 3))
ax1.tick_params(axis='x', labelsize=8)  # Adjust x-axis tick font size
ax1.tick_params(axis='y', labelsize=8)  # Adjust y-axis tick font size if needed
ax1.set_xlabel(f"Gradients\n{corr_text}", fontsize=11)
ax1.set_ylabel(f"Post-treatment log-fold change\n(expanded vs. non-expanded)", fontsize=11)
fig.tight_layout()
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_post_by_gradient.png', dpi=300)
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_post_by_gradient.svg', format='svg')
fig.set_size_inches(old_fig_size)

## (2.1) lfc_post vs. salient_genes_ranks
fig.set_size_inches([10, 5])
fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(diffexp_df['salient_genes_rank'], diffexp_df['lfc_post'], s=5, alpha=0.8)
ax1.set_xlabel('Gradient ranks')
ax1.set_ylabel('Log-fold change\n(expanded vs. non-expanded post-treatment)')
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_post_by_gradient_rank.png')
fig.set_size_inches(old_fig_size)

## (3) lfc_pre vs. salient_genes
fig.set_size_inches([3, 5])
fig.clf()
ax1 = fig.subplots(1, 1)
x = diffexp_df['salient_genes']
y = diffexp_df['lfc_pre']
ax1.scatter(x, y, s=3, alpha=0.8)
corr, p_value = pearsonr(x, y)
if p_value == 0.00:
    formatted_pval = '0.00'
else:
    formatted_pval = f"10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}" if f'{p_value:.1e}'.startswith('1e') else f"{float(f'{p_value:.1e}'.split('e')[0])} \\times 10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}"
corr_text = f"$r = {corr:.2f}, p = {formatted_pval}$"
# ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, fontsize=9,
#          verticalalignment='top')
# Fit line using linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept
ax1.plot(line_x, line_y, color='red', linestyle='-', linewidth=1, label='Fitted Line')

ax1.set_xticks(np.linspace(-0.0002, 0.0002, 3))
ax1.tick_params(axis='x', labelsize=8)  # Adjust x-axis tick font size
ax1.tick_params(axis='y', labelsize=8)  # Adjust y-axis tick font size if needed
ax1.set_xlabel(f"Gradients\n{corr_text}", fontsize=11)
ax1.set_ylabel(f'Pre-treatment log-fold change\n(expanding vs. non-expanding)', fontsize=11)
fig.tight_layout()
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_pre_by_gradient.png', dpi=300)
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_pre_by_gradient.svg', format='svg')
fig.set_size_inches(old_fig_size)

## (4.1) lfc_expanded vs. salient_genes
# fig.set_size_inches([10, 5])
# fig.clf()
# ax1 = fig.subplots(1, 1)
# ax1.scatter(diffexp_df['salient_genes'], diffexp_df['lfc_expanded'], s=5, alpha=0.8)
# ax1.set_xlabel('Gradients')
# ax1.set_ylabel('Log-fold change\n(Going-to-be-expanded vs. expanded)')
# fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{cellType}/pairwise_comparison/lfc_expanded_by_gradient.png')
# fig.set_size_inches(old_fig_size)

## (4.2) lfc_nonexpanded vs. salient_genes
# fig.set_size_inches([10, 5])
# fig.clf()
# ax1 = fig.subplots(1, 1)
# ax1.scatter(diffexp_df['salient_genes'], diffexp_df['lfc_nonexpanded'], s=5, alpha=0.8)
# ax1.set_xlabel('Gradients')
# ax1.set_ylabel('Log-fold change\n(Not going-to-be-expanded vs. non-expanded)')
# fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{cellType}/pairwise_comparison/lfc_nonexpanded_by_gradient.png')
# fig.set_size_inches(old_fig_size)

## (5) lfc_clone_pre_10 vs. salient_genes
fig.set_size_inches([3, 5])
fig.clf()
ax1 = fig.subplots(1, 1)
x = diffexp_df['salient_genes']
y = diffexp_df['lfc_clone_pre_10']
ax1.scatter(x, y, s=5, alpha=0.8)
corr, p_value = pearsonr(x, y)
if p_value == 0.00:
    formatted_pval = '0.00'
else:
    formatted_pval = f"10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}" if f'{p_value:.1e}'.startswith('1e') else f"{float(f'{p_value:.1e}'.split('e')[0])} \\times 10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}"
corr_text = f"$r = {corr:.2f}, p = {formatted_pval}$"
# ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, fontsize=9,
#          verticalalignment='top')
# Fit line using linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept
ax1.plot(line_x, line_y, color='red', linestyle='-', linewidth=1, label='Fitted Line')

ax1.set_xticks(np.linspace(-0.0002, 0.0002, 3))
ax1.tick_params(axis='x', labelsize=8)  # Adjust x-axis tick font size
ax1.tick_params(axis='y', labelsize=8)  # Adjust y-axis tick font size if needed
ax1.set_xlabel(f"Gradients\n{corr_text}", fontsize=11)
ax1.set_ylabel(f'Pre-treatment log-fold change\n(clone size > 10 vs. <=10)', fontsize=11)
fig.tight_layout()
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_clone_pre_10_by_gradient.png', dpi=300)
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_clone_pre_10_by_gradient.svg', format='svg')
fig.set_size_inches(old_fig_size)

## (6) lfc_clone_pre_1 vs. salient_genes
fig.set_size_inches([3, 5])
fig.clf()
ax1 = fig.subplots(1, 1)
x = diffexp_df['salient_genes']
y = diffexp_df['lfc_clone_pre_1']
ax1.scatter(x, y, s=5, alpha=0.8)
corr, p_value = pearsonr(x, y)
if p_value == 0.00:
    formatted_pval = '0.00'
else:
    formatted_pval = f"10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}" if f'{p_value:.1e}'.startswith('1e') else f"{float(f'{p_value:.1e}'.split('e')[0])} \\times 10^{{{int(f'{p_value:.1e}'.split('e')[1])}}}"
corr_text = f"$r = {corr:.2f}, p = {formatted_pval}$"
# ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, fontsize=9,
#          verticalalignment='top')
# Fit line using linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept
ax1.plot(line_x, line_y, color='red', linestyle='-', linewidth=1, label='Fitted Line')

ax1.set_xticks(np.linspace(-0.0002, 0.0002, 3))
ax1.tick_params(axis='x', labelsize=8)  # Adjust x-axis tick font size
ax1.tick_params(axis='y', labelsize=8)  # Adjust y-axis tick font size if needed
ax1.set_xlabel(f"Gradients\n{corr_text}", fontsize=11)
ax1.set_ylabel(f'Pre-treatment log-fold change\n(clone size > 1 vs. <=1)', fontsize=11)
fig.tight_layout()
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_clone_pre_1_by_gradient.png', dpi=300)
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_clone_pre_1_by_gradient.svg', format='svg')
fig.set_size_inches(old_fig_size)

# fig.set_size_inches([10, 5])
# fig.clf()
# ax1 = fig.subplots(1, 1)
# ax1.scatter(diffexp_df['salient_genes'], diffexp_df['lfc_clone_pre_1'], s=5, alpha=0.8)
# ax1.set_xlabel('Gradients')
# ax1.set_ylabel('Log-fold change\n(clone size > 1 vs. <= 1 pre-treatment)')
# fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{cellType}/pairwise_comparison/lfc_clone_pre_1_by_gradient.png')
# fig.set_size_inches(old_fig_size)

## (6) lfc_post vs. lfc_expanded
# fig.set_size_inches([8, 6])
# fig.clf()
# ax1 = fig.subplots(1, 1)
# ax1.scatter(diffexp_df['lfc_post'], diffexp_df['lfc_expanded'], s=5, alpha=0.8)
# ax1.set_xlabel('Log-fold change\n(expanded vs. non-expanded post-treatment)')
# ax1.set_ylabel('Log-fold change\n(expanded vs. going-to-expand)')
# fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{cellType}/pairwise_comparison/lfc_post_by_lfc_expand.png')
# fig.set_size_inches(old_fig_size)

## (7) lfc_post vs. lfc_pre
fig.set_size_inches([8, 6])
fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(diffexp_df['lfc_post'], diffexp_df['lfc_pre'], s=5, alpha=0.8)
ax1.set_xlabel('Log-fold change\n(expanded vs. non-expanded post-treatment)')
ax1.set_ylabel('Log-fold change\n(going-to-expand vs. going-to-nonexpand)')
fig.savefig(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/pairwise_comparison/lfc_post_by_lfc_pre.png')
fig.set_size_inches(old_fig_size)


################################
##### Analysis of the top genes
################################

### lfc_post_by_gradient --------------------------------------------------------------------

# Define thresholds
lfc_type = 'lfc_post'
lfc_title = 'Log Fold Change\n(expanded vs. non-expanded post-treatment)'
lfc_high_threshold = diffexp_df['lfc_post'].quantile(0.99)
lfc_low_threshold = diffexp_df['lfc_post'].quantile(0.01)
grad_high_threshold = diffexp_df['salient_genes'].quantile(0.99)
grad_low_threshold = diffexp_df['salient_genes'].quantile(0.01)
print(lfc_high_threshold, lfc_low_threshold, grad_high_threshold, grad_low_threshold)

## (1) High log-fold change and high gradient
colname = 'high_lfc_high_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    (diffexp_df['lfc_post'] >= lfc_high_threshold) & 
    (diffexp_df['salient_genes'] >= grad_high_threshold), 
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

## (2) Low log-fold change and low gradient
colname = 'low_lfc_low_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    (diffexp_df['lfc_post'] <= lfc_low_threshold) & 
    (diffexp_df['salient_genes'] <= grad_low_threshold), 
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

## (3) Interesting one: around 0 log-fold change but high gradient
top_n = 1000
post_high_rank = diffexp_df['lfc_post_rank'].max() - top_n
post_low_rank = top_n
colname = 'zero_lfc_high_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    ((diffexp_df['lfc_post_rank'] >= post_low_rank) & 
     (diffexp_df['lfc_post_rank'] <= post_high_rank)) & 
    (diffexp_df['salient_genes'] >= grad_high_threshold),
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

### lfc_clone_pre_1_by_gradient --------------------------------------------------------------------

# Define thresholds
lfc_type = 'lfc_clone_pre_1'
lfc_title = 'Log Fold Change\n(clone size > 1 vs. <= 1 pre-treatment)'
lfc_high_threshold = diffexp_df[lfc_type].quantile(0.99)
lfc_low_threshold = diffexp_df[lfc_type].quantile(0.01)
grad_high_threshold = diffexp_df['salient_genes'].quantile(0.99)
grad_low_threshold = diffexp_df['salient_genes'].quantile(0.01)
print(lfc_high_threshold, lfc_low_threshold, grad_high_threshold, grad_low_threshold)

## (1) High log-fold change and high gradient
colname = 'high_lfc_high_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    (diffexp_df[lfc_type] >= lfc_high_threshold) & 
    (diffexp_df['salient_genes'] >= grad_high_threshold), 
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

## (2) Low log-fold change and low gradient
colname = 'low_lfc_low_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    (diffexp_df[lfc_type] <= lfc_low_threshold) & 
    (diffexp_df['salient_genes'] <= grad_low_threshold), 
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

## (3) Interesting one: around 0 log-fold change but high gradient
top_n = 1000
post_high_rank = diffexp_df[f'{lfc_type}_rank'].max() - top_n
post_low_rank = top_n
colname = 'zero_lfc_high_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    ((diffexp_df[f'{lfc_type}_rank'] >= post_low_rank) & 
     (diffexp_df[f'{lfc_type}_rank'] <= post_high_rank)) & 
    (diffexp_df['salient_genes'] >= grad_high_threshold),
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

### lfc_clone_pre_10_by_gradient --------------------------------------------------------------------

# Define thresholds
lfc_type = 'lfc_clone_pre_10'
lfc_title = 'Log Fold Change\n(clone size > 10 vs. <= 10 pre-treatment)'
lfc_high_threshold = diffexp_df[lfc_type].quantile(0.99)
lfc_low_threshold = diffexp_df[lfc_type].quantile(0.01)
grad_high_threshold = diffexp_df['salient_genes'].quantile(0.99)
grad_low_threshold = diffexp_df['salient_genes'].quantile(0.01)
print(lfc_high_threshold, lfc_low_threshold, grad_high_threshold, grad_low_threshold)

## (1) High log-fold change and high gradient
colname = 'high_lfc_high_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    (diffexp_df[lfc_type] >= lfc_high_threshold) & 
    (diffexp_df['salient_genes'] >= grad_high_threshold), 
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

## (2) Low log-fold change and low gradient
colname = 'low_lfc_low_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    (diffexp_df[lfc_type] <= lfc_low_threshold) & 
    (diffexp_df['salient_genes'] <= grad_low_threshold), 
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)

## (3) Interesting one: around 0 log-fold change but high gradient
top_n = 1000
post_high_rank = diffexp_df[f'{lfc_type}_rank'].max() - top_n
post_low_rank = top_n
colname = 'zero_lfc_high_grad'
diffexp_df[colname] = 'No'  # Default category
diffexp_df.loc[
    ((diffexp_df[f'{lfc_type}_rank'] >= post_low_rank) & 
     (diffexp_df[f'{lfc_type}_rank'] <= post_high_rank)) & 
    (diffexp_df['salient_genes'] >= grad_high_threshold),
    colname
] = 'Yes'
investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title, de_type)


### lfc_expanded_by_gradient --------------------------------------------------------------------

# # Define thresholds
# lfc_type = 'lfc_expanded'
# lfc_title = 'Log Fold Change\n(expanded vs. going-to-expand)'
# lfc_high_threshold = diffexp_df[lfc_type].quantile(0.99)
# lfc_low_threshold = diffexp_df[lfc_type].quantile(0.01)
# grad_high_threshold = diffexp_df['salient_genes'].quantile(0.99)
# grad_low_threshold = diffexp_df['salient_genes'].quantile(0.01)
# print(lfc_high_threshold, lfc_low_threshold, grad_high_threshold, grad_low_threshold)

# ## (1) low log-fold change and high gradient
# colname = 'low_lfc_high_grad'
# diffexp_df[colname] = 'No'  # Default category
# diffexp_df.loc[
#     (diffexp_df[lfc_type] <= lfc_low_threshold) & 
#     (diffexp_df['salient_genes'] >= grad_high_threshold), 
#     colname
# ] = 'Yes'
# investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title)

# ## (2) high log-fold change and low gradient
# colname = 'high_lfc_low_grad'
# diffexp_df[colname] = 'No'  # Default category
# diffexp_df.loc[
#     (diffexp_df[lfc_type] >= lfc_high_threshold) & 
#     (diffexp_df['salient_genes'] <= grad_low_threshold), 
#     colname
# ] = 'Yes'
# investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title)

# ## (3) Interesting one: around 0 log-fold change but high gradient
# top_n = 1000
# post_high_rank = diffexp_df[f'{lfc_type}_rank'].max() - top_n
# post_low_rank = top_n
# colname = 'zero_lfc_high_grad'
# diffexp_df[colname] = 'No'  # Default category
# diffexp_df.loc[
#     ((diffexp_df[f'{lfc_type}_rank'] >= post_low_rank) & 
#      (diffexp_df[f'{lfc_type}_rank'] <= post_high_rank)) & 
#     (diffexp_df['salient_genes'] >= grad_high_threshold),
#     colname
# ] = 'Yes'
# investigate_corner_genes(diffexp_df, colname, cellType, lfc_type, lfc_title)

