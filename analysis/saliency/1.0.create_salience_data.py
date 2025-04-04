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
from saliency_helpers import investigate_corner_genes
import pyreadr
import pyarrow
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

with open('/home/che/TRIM/data_processed/data_rna.npz', 'rb') as f:
    npzfile = np.load(f)
    x_rna_norm = npzfile['data_rna']

with open('/home/che/TRIM/data_processed/data_rna_counts.pkl', 'rb') as f:
    x_rna_counts = pickle.load(f)

with open('/home/che/TRIM/data_processed/data_tcr.npz', 'rb') as f:
    npzfile = np.load(f)
    x_tcr = npzfile['data_tcr']

with open('/home/che/TRIM/data_processed/data_labels.npz', 'rb') as f:
    npzfile = np.load(f)
    x_label = npzfile['data_labels']

with open('/home/che/TRIM/data_processed/data_all_tcrs.npz', 'rb') as f:
    npzfile = np.load(f, allow_pickle=True)
    df_all_tcrs = npzfile['data_all_tcrs']
    rows = npzfile['rows']
    cols = npzfile['cols']
    df_all_tcrs = pd.DataFrame(df_all_tcrs, index=rows, columns=cols)

with open('/home/che/TRIM/data_processed/salience_grads_new.npz', 'rb') as f: 
    npzfile = np.load(f, allow_pickle=True) 
    out_grads = npzfile['out_grads'] 

with open('/home/che/TRIM/data_processed/combined_data_columns.npz', 'rb') as f:
    npzfile = np.load(f, allow_pickle=True)
    combined_data_columns = npzfile['cols']

###############
############### END data loading
############### 


###############
############### Use scanpy to library-normalize the data
############### 

adata_rna = AnnData(x_rna_counts)
adata_rna.var_names = combined_data_columns
adata_rna.var['gene_names'] = combined_data_columns
# normalize to median total counts (library size normalization)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
x_rna = adata_rna.X.copy() # in normalized counts
x_rna.sum(axis=1) # similar total counts for each cell
# log1p transform
sc.pp.log1p(adata_rna)
###############
############### End checks
############### 

def check(gene):
    if (x_rna[blood_expanded == 0][:, gene] > 0).sum() == 0:
        return False
    if (x_rna[blood_expanded == 1][:, gene] > 0).sum() == 0:
        return False
    return True

# Define expanded and non-expanded blood TCRs ------------------------------------------------
blood_exists_pre_and_post = np.logical_and(df_all_tcrs.fillna(0).iloc[:, 0] > 0, df_all_tcrs.fillna(0).iloc[:, 1] > 0) #have TCR counts before and after treatment
blood_exists_pre_or_post = np.logical_or(df_all_tcrs.fillna(0).iloc[:, 0] > 0, df_all_tcrs.fillna(0).iloc[:, 1] > 0) #have TCR counts before or after treatment
blood_exists_pre = df_all_tcrs.fillna(0).iloc[:, 0] > 0 #have TCR counts before treatment

# (1) This is the situation where we have all pre- and post-treatment TCRs in blood 
if de_type == 'include_emergent_disappear':
    blood_expanded = np.logical_and(blood_exists_pre_or_post, df_all_tcrs.fillna(0).iloc[:, 0] < df_all_tcrs.fillna(0).iloc[:, 1]) # have more counts after in blood than before
    blood_nonexpanded = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] >= df_all_tcrs.fillna(0).iloc[:, 1]) #have counts before >= after in blood
    print('Number of blood expanded:', blood_expanded.sum(), 'Number of blood non-expanded:', blood_nonexpanded.sum()) # 12326, 14962
elif de_type == 'no_emergent_include_disappear':
    blood_expanded = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] < df_all_tcrs.fillna(0).iloc[:, 1])
    blood_nonexpanded = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] >= df_all_tcrs.fillna(0).iloc[:, 1])
    print('Number of blood expanded:', blood_expanded.sum(), 'Number of blood non-expanded:', blood_nonexpanded.sum()) # 185, 14962
elif de_type == 'no_emergent_no_disappear':
    blood_expanded = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] < df_all_tcrs.fillna(0).iloc[:, 1])
    blood_nonexpanded = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] >= df_all_tcrs.fillna(0).iloc[:, 1])
    print('Number of blood expanded:', blood_expanded.sum(), 'Number of blood non-expanded:', blood_nonexpanded.sum()) # 185, 540

df_all_tcrs['blood_expanded'] = blood_expanded
df_all_tcrs['blood_nonexpanded'] = blood_nonexpanded
assert (df_all_tcrs.loc[blood_expanded, ['1', '2']].isna().sum(axis=1) < 2).all() # check that blood_expanded TCRs have nonNA values in either columns 0 or 1
assert (df_all_tcrs.loc[blood_nonexpanded, ['1', '2']].isna().sum(axis=1) < 2).all() # check that blood_expanded TCRs have nonNA values in either columns 0 or 1
blood_tcr_expanded_idx_list = np.where(blood_expanded.values == True)[0].tolist()
blood_tcr_nonexpanded_idx_list = np.where(blood_nonexpanded.values == True)[0].tolist()
print('Number of blood expanded:', len(blood_tcr_expanded_idx_list), 'Number of blood non-expanded:', len(blood_tcr_nonexpanded_idx_list))
# include_emergent_disappear: 12326, 14962
# no_emergent_include_disappear: 185, 14962
# no_emergent_no_disappear: 185, 540

assert len(set(blood_tcr_expanded_idx_list).intersection(set(blood_tcr_nonexpanded_idx_list))) == 0, "Some TCRs are both expanded and non-expanded"

# Define big and small clone TCRs pre-treatment in blood cells -----------------------------------

if de_type == 'include_emergent_disappear' or de_type == 'no_emergent_include_disappear':
    blood_big_clone_pre_10 = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] > 10) #have more than 10 counts before in blood
    blood_small_clone_pre_10 = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] <= 10) #have 10 or less counts before in blood
    blood_big_clone_pre_1 = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] > 1) #have more than 1 count before in blood
    blood_small_clone_pre_1 = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] <= 1) #have 1 or less count before in blood
    print('Number of blood big clone pre-10:', blood_big_clone_pre_10.sum(), 'Number of blood small clone pre-10:', blood_small_clone_pre_10.sum()) # 56, 15091
    print('Number of blood big clone pre-1:', blood_big_clone_pre_1.sum(), 'Number of blood small clone pre-1:', blood_small_clone_pre_1.sum()) # 671, 14476
    blood_tcr_exists_idx_list = np.where(blood_exists_pre.values == True)[0].tolist()
elif de_type == 'no_emergent_no_disappear':
    blood_big_clone_pre_10 = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] > 10) #have more than 10 counts before in blood
    blood_small_clone_pre_10 = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] <= 10) #have 10 or less counts before in blood
    blood_big_clone_pre_1 = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] > 1) #have more than 1 count before in blood
    blood_small_clone_pre_1 = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] <= 1) #have 1 or less count before in blood
    print('Number of blood big clone pre-10:', blood_big_clone_pre_10.sum(), 'Number of blood small clone pre-10:', blood_small_clone_pre_10.sum()) # 43, 682
    print('Number of blood big clone pre-1:', blood_big_clone_pre_1.sum(), 'Number of blood small clone pre-1:', blood_small_clone_pre_1.sum()) # 349, 376
    blood_tcr_exists_idx_list = np.where(blood_exists_pre_and_post.values == True)[0].tolist()

df_all_tcrs['blood_big_clone_pre_10'] = blood_big_clone_pre_10
df_all_tcrs['blood_small_clone_pre_10'] = blood_small_clone_pre_10
df_all_tcrs['blood_big_clone_pre_1'] = blood_big_clone_pre_1
df_all_tcrs['blood_small_clone_pre_1'] = blood_small_clone_pre_1

blood_tcr_big_clone_pre_10_idx_list = np.where(blood_big_clone_pre_10.values == True)[0].tolist()
blood_tcr_small_clone_pre_10_idx_list = np.where(blood_small_clone_pre_10.values == True)[0].tolist()
blood_tcr_big_clone_pre_1_idx_list = np.where(blood_big_clone_pre_1.values == True)[0].tolist()
blood_tcr_small_clone_pre_1_idx_list = np.where(blood_small_clone_pre_1.values == True)[0].tolist()
print('Number of blood big clone pre-10:', len(blood_tcr_big_clone_pre_10_idx_list), 'Number of blood small clone pre-10:', len(blood_tcr_small_clone_pre_10_idx_list)) 
# Include disappear: 56, 15091 
# Exlucde disappear: 43, 682
print('Number of blood big clone pre-1:', len(blood_tcr_big_clone_pre_1_idx_list), 'Number of blood small clone pre-1:', len(blood_tcr_small_clone_pre_1_idx_list)) 
# Include disappear: 671, 14476
# Exlucde disappear: 349, 376

assert set(blood_tcr_big_clone_pre_10_idx_list).intersection(set(blood_tcr_small_clone_pre_10_idx_list)) == set(), "Some TCRs are both big and small clone"
assert set(blood_tcr_big_clone_pre_1_idx_list).intersection(set(blood_tcr_small_clone_pre_1_idx_list)) == set(), "Some TCRs are both big and small clone"
assert set(blood_tcr_big_clone_pre_10_idx_list) - set(blood_tcr_big_clone_pre_1_idx_list) == set(), "Some TCRs are big clone pre-1 but not big clone pre-10"
assert set(blood_tcr_big_clone_pre_10_idx_list).union(set(blood_tcr_small_clone_pre_10_idx_list)) == set(blood_tcr_exists_idx_list), "Some TCRs are missing"
assert set(blood_tcr_big_clone_pre_1_idx_list).union(set(blood_tcr_small_clone_pre_1_idx_list)) == set(blood_tcr_exists_idx_list), "Some TCRs are missing"

# Define expanding and non-expanding blood TCRs for pre-treatment measurements ------------------------------------------------
if de_type == 'include_emergent_disappear' or de_type == 'no_emergent_include_disappear':
    blood_expanding = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] < df_all_tcrs.fillna(0).iloc[:, 1]) # have more counts after in blood than before (including emergent)
    blood_nonexpanding = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] >= df_all_tcrs.fillna(0).iloc[:, 1]) #have counts before >= after in blood
    print('Number of blood expanding:', blood_expanding.sum(), 'Number of blood non-expanding:', blood_nonexpanding.sum()) # 185, 14962
elif de_type == 'no_emergent_no_disappear':
    blood_expanding = np.logical_and(blood_exists_pre, df_all_tcrs.fillna(0).iloc[:, 0] < df_all_tcrs.fillna(0).iloc[:, 1]) # have more counts after in blood than before (including emergent)
    blood_nonexpanding = np.logical_and(blood_exists_pre_and_post, df_all_tcrs.fillna(0).iloc[:, 0] >= df_all_tcrs.fillna(0).iloc[:, 1]) #have counts before >= after in blood
    print('Number of blood expanding:', blood_expanding.sum(), 'Number of blood non-expanding:', blood_nonexpanding.sum()) # 185, 540

df_all_tcrs['blood_expanding'] = blood_expanding
df_all_tcrs['blood_nonexpanding'] = blood_nonexpanding
assert (df_all_tcrs.loc[blood_expanding, ['1']].isna().sum(axis=1) == 0).all() # check that blood_expanding TCRs have nonNA values in column 0
assert (df_all_tcrs.loc[blood_nonexpanding, ['1']].isna().sum(axis=1) == 0).all() # check that blood_expanding TCRs have nonNA values in column 0
blood_tcr_expanding_idx_list = np.where(blood_expanding.values == True)[0].tolist()
blood_tcr_nonexpanding_idx_list = np.where(blood_nonexpanding.values == True)[0].tolist()
print('Number of blood expanding:', len(blood_tcr_expanding_idx_list), 'Number of blood non-expanding:', len(blood_tcr_nonexpanding_idx_list))
# Include disappear: 185, 14962
# Exclude disappear: 185, 540

assert set(blood_tcr_expanding_idx_list).union(set(blood_tcr_nonexpanding_idx_list)) == set(blood_tcr_exists_idx_list), "Some TCRs are missing"

###############
############### Get correct cell types
###############

col_bloodtumor, col_prepost, col_celltype, col_patient, col_tcr, col_tcr_v, col_tcr_j, col_treatment = list(range(x_label.shape[1]))

# blood_expanded_tcr indicates whether the cell has expanded TCRs post-treatment
blood_expanded_tcr = np.take(blood_expanded.values, x_label[:, col_tcr].astype(np.int32), axis=0)
blood_nonexpanded_tcr = np.take(blood_nonexpanded.values, x_label[:, col_tcr].astype(np.int32), axis=0)
for idx in blood_tcr_expanded_idx_list:
    condition = np.where(x_label[:, col_tcr].astype(np.int32) == idx)
    assert (blood_expanded_tcr[condition]).all(), f"Assertion failed for idx {idx}"
for idx in blood_tcr_nonexpanded_idx_list:
    condition = np.where(x_label[:, col_tcr].astype(np.int32) == idx)
    assert (blood_nonexpanded_tcr[condition]).all(), f"Assertion failed for idx {idx}"
print(blood_expanded_tcr.sum(), blood_nonexpanded_tcr.sum()) 
# emergent and disappear: 16538, 27654 
# No emergent and no disappear: 3258, 11220
# No emergent and include disappear: 3258, 27654

# clone size flag for cells pre-treatment
big_clone_pre_10_tcr = np.take(blood_big_clone_pre_10.values, x_label[:, col_tcr].astype(np.int32), axis=0)
small_clone_pre_10_tcr = np.take(blood_small_clone_pre_10.values, x_label[:, col_tcr].astype(np.int32), axis=0)
big_clone_pre_1_tcr = np.take(blood_big_clone_pre_1.values, x_label[:, col_tcr].astype(np.int32), axis=0)
small_clone_pre_1_tcr = np.take(blood_small_clone_pre_1.values, x_label[:, col_tcr].astype(np.int32), axis=0)
print(big_clone_pre_10_tcr.sum(), small_clone_pre_10_tcr.sum()) 
# Include disappear: 10689, 20223
# Exclude disappear: 10228, 4250
print(big_clone_pre_1_tcr.sum(), small_clone_pre_1_tcr.sum()) 
# Include disappear: 14805, 16107 
# Exclude disappear: 13240, 1238

# expanding and non-expanding cell flags
blood_expanding_tcr = np.take(blood_expanding.values, x_label[:, col_tcr].astype(np.int32), axis=0)
blood_nonexpanding_tcr = np.take(blood_nonexpanding.values, x_label[:, col_tcr].astype(np.int32), axis=0)
for idx in blood_tcr_expanding_idx_list:
    condition = np.where(x_label[:, col_tcr].astype(np.int32) == idx)
    assert (blood_expanding_tcr[condition]).all(), f"Assertion failed for idx {idx}"
for idx in blood_tcr_nonexpanding_idx_list:
    condition = np.where(x_label[:, col_tcr].astype(np.int32) == idx)
    assert (blood_nonexpanding_tcr[condition]).all(), f"Assertion failed for idx {idx}"
print(blood_expanding_tcr.sum(), blood_nonexpanding_tcr.sum()) 
# Include disappear: 3258, 27654
# Exclude disappear: 3258, 11220

# flag expanded and non-expanded cells post-treatment -------------------
blood_expanded_post = np.logical_and(blood_expanded_tcr, x_label[:, col_prepost] == 1) #post-treatment
print('blood_expanded_post:', blood_expanded_post.sum()) 
# Include emergent: 15303
# Exclude emergent: 2175
blood_nonexpanded_post = np.logical_and(blood_nonexpanded_tcr, x_label[:, col_prepost] == 1) #post-treatment
print('blood_nonexpanded_post:', blood_nonexpanded_post.sum()) 
# Include disappear: 6412
# Exclude disappear: 5656

# flag expanding and non-expanding cells pre-treatment -------------------
blood_expanding_pre = np.logical_and(blood_expanding_tcr, x_label[:, col_prepost] == 0) #pre-treatment
print('blood_expanding_pre:', blood_expanding_pre.sum()) 
# In all cases: 1083

blood_nonexpanding_pre = np.logical_and(blood_nonexpanding_tcr, x_label[:, col_prepost] == 0) #pre-treatment
print('blood_nonexpanding_pre:', blood_nonexpanding_pre.sum()) 
# Include disappear: 21242
# Exclude disappear: 5564

# flag cells of different clone-size in pre-treatment ------------------------------------------------
big_clone_pre_10 = np.logical_and(big_clone_pre_10_tcr, x_label[:, col_prepost] == 0) #pre-treatment
small_clone_pre_10 = np.logical_and(small_clone_pre_10_tcr, x_label[:, col_prepost] == 0) #pre-treatment
print('big_clone_pre_10:', big_clone_pre_10.sum(), 'small_clone_pre_10:', small_clone_pre_10.sum()) 
# Include disappear: 5459, 16866
# Exclude disappear: 5033, 1614
big_clone_pre_1 = np.logical_and(big_clone_pre_1_tcr, x_label[:, col_prepost] == 0) #pre-treatment
small_clone_pre_1 = np.logical_and(small_clone_pre_1_tcr, x_label[:, col_prepost] == 0) #pre-treatment
print('big_clone_pre_1:', big_clone_pre_1.sum(), 'small_clone_pre_1:', small_clone_pre_1.sum()) 
# Include disappear: 7504, 14821
# Exclude disappear: 6188, 459

# Select correct cell types --------------------------------------------------------------------
blood_cells = x_label[:, col_bloodtumor] == 0  # restrict to blood cells
print('blood_cells:', blood_cells.sum()) # 34890
cellType = 'CD8' # CD4, CD8
if cellType == 'CD4':
    cell_type_flag = np.logical_and(blood_cells, x_label[:, col_celltype] == 0) # cd4 blood
    print('CD4 blood', cell_type_flag.sum())
elif cellType == 'CD8':
    cell_type_flag = np.logical_and(blood_cells, x_label[:, col_celltype] == 1) # cd8 blood
    print('CD8 blood', cell_type_flag.sum()) # 10704

# restrict to desired cell types (blood) --------------------------------------------------------------------
blood_expanded_post = np.logical_and(blood_expanded_post, cell_type_flag)
print('blood_expanded_post:', blood_expanded_post.sum()) # Include emergent: 3688, Exclude emergent: 1595
blood_nonexpanded_post = np.logical_and(blood_nonexpanded_post, cell_type_flag)
print('blood_nonexpanded_post:', blood_nonexpanded_post.sum()) # Include disappear: 1306, Exclude disappear: 1306
blood_expanding_pre = np.logical_and(blood_expanding_pre, cell_type_flag)
print('blood_expanded_pre:', blood_expanding_pre.sum()) # In all cases: 991
blood_nonexpanding_pre = np.logical_and(blood_nonexpanding_pre, cell_type_flag)
print('blood_nonexpanding_pre:', blood_nonexpanding_pre.sum()) # Include disappear: 4719, Exclude disappear: 1779
assert blood_expanded_post.sum() > blood_expanding_pre.sum(), "Assertion failed for expanded cells"
assert blood_nonexpanded_post.sum() <= blood_nonexpanding_pre.sum(), "Assertion failed for non-expanded cells"

big_clone_pre_10 = np.logical_and(big_clone_pre_10, cell_type_flag)
print('big_clone_pre_10:', big_clone_pre_10.sum()) # Include disappear: 1927, Exclude disappear: 1561
small_clone_pre_10 = np.logical_and(small_clone_pre_10, cell_type_flag)
print('small_clone_pre_10:', small_clone_pre_10.sum()) # Include disappear:  3783, Exclude disappear: 1209
big_clone_pre_1 = np.logical_and(big_clone_pre_1, cell_type_flag)
print('big_clone_pre_1:', big_clone_pre_1.sum()) # Include disappear: 3428, Exclude disappear: 2587
small_clone_pre_1 = np.logical_and(small_clone_pre_1, cell_type_flag)
print('small_clone_pre_1:', small_clone_pre_1.sum()) # Include disappear: 2282, Exclude disappear: 183

assert small_clone_pre_10.sum() > small_clone_pre_1.sum(), "Assertion failed for big clone pre-10"
assert big_clone_pre_1.sum() > big_clone_pre_10.sum(), "Assertion failed for big clone pre-1"

### Export the data ------------------------------------------------
genes_filters = (x_rna > 0).sum(axis=0) > 10 # Filter genes expressed in more than 10 cells
print('Number of genes:', genes_filters.sum()) # 19256

data_folder_path = f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/data'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
seurat_folder_path = f'{data_folder_path}/seurat/data'
if not os.path.exists(seurat_folder_path):
    os.makedirs(seurat_folder_path)

## Export adata_rna into h5ad format
adata_rna.write(f'{data_folder_path}/adata_normalized.h5ad')
print('adata_rna.shape', adata_rna.shape)

## Export indicators into pickle file
with open(f'{data_folder_path}/indicators.pkl', 'wb') as f:
    pickle.dump({'blood_expanded_post': blood_expanded_post, 
                 'blood_nonexpanded_post': blood_nonexpanded_post, 
                 'blood_expanding_pre': blood_expanding_pre, 
                 'blood_nonexpanding_pre': blood_nonexpanding_pre, 
                 'big_clone_pre_10': big_clone_pre_10, 
                 'small_clone_pre_10': small_clone_pre_10, 
                 'big_clone_pre_1': big_clone_pre_1, 
                 'small_clone_pre_1': small_clone_pre_1}, f)

## Export raw gene counts data (lfc_post = post_expanded - post_nonexpanded)
x_rna_counts_lfc_post = x_rna_counts.copy()
# subset columns using genes_filters
x_rna_counts_lfc_post = x_rna_counts_lfc_post.iloc[:, genes_filters]
x_rna_counts_lfc_post['blood_expanded_post'] = blood_expanded_post
x_rna_counts_lfc_post['blood_nonexpanded_post'] = blood_nonexpanded_post
x_rna_counts_lfc_post['blood_post'] = np.where(blood_expanded_post, 'expanded', np.where(blood_nonexpanded_post, 'non-expanded', 'NA'))
assert (x_rna_counts_lfc_post['blood_post'] == 'expanded').sum() == (x_rna_counts_lfc_post['blood_expanded_post']).sum(), "Assertion failed for expanded cells"
assert (x_rna_counts_lfc_post['blood_post'] == 'non-expanded').sum() == (x_rna_counts_lfc_post['blood_nonexpanded_post']).sum(), "Assertion failed for non-expanded cells"
x_rna_counts_lfc_post = x_rna_counts_lfc_post[x_rna_counts_lfc_post['blood_post'] != 'NA']
print('x_rna_counts_lfc_post.shape', x_rna_counts_lfc_post.shape) # 4994, 19259
x_rna_counts_lfc_post.to_parquet(f"{data_folder_path}/seurat/data/x_rna_counts_lfc_post.parquet", engine='pyarrow')

## Export raw gene counts data (lfc_pre = pre_expanded - pre_nonexpanded)
x_rna_counts_lfc_pre = x_rna_counts.copy()
x_rna_counts_lfc_pre = x_rna_counts_lfc_pre.iloc[:, genes_filters]
x_rna_counts_lfc_pre['blood_expanding_pre'] = blood_expanding_pre
x_rna_counts_lfc_pre['blood_nonexpanding_pre'] = blood_nonexpanding_pre
x_rna_counts_lfc_pre['blood_pre'] = np.where(blood_expanding_pre, 'expanding', np.where(blood_nonexpanding_pre, 'non-expanding', 'NA'))
assert (x_rna_counts_lfc_pre['blood_pre'] == 'expanding').sum() == (x_rna_counts_lfc_pre['blood_expanding_pre']).sum(), "Assertion failed for expanded cells"
assert (x_rna_counts_lfc_pre['blood_pre'] == 'non-expanding').sum() == (x_rna_counts_lfc_pre['blood_nonexpanding_pre']).sum(), "Assertion failed for non-expanded cells"
x_rna_counts_lfc_pre = x_rna_counts_lfc_pre[x_rna_counts_lfc_pre['blood_pre'] != 'NA']
print('x_rna_counts_lfc_pre.shape', x_rna_counts_lfc_pre.shape) # 5710, 19259
x_rna_counts_lfc_pre.to_parquet(f"{data_folder_path}/seurat/data/x_rna_counts_lfc_pre.parquet", engine='pyarrow')

## Export raw gene counts data (lfc_clone_pre_10 = big_clone_pre_10 - small_clone_pre_10)
x_rna_counts_lfc_clone_pre_10 = x_rna_counts.copy()
x_rna_counts_lfc_clone_pre_10 = x_rna_counts_lfc_clone_pre_10.iloc[:, genes_filters]
x_rna_counts_lfc_clone_pre_10['big_clone_pre_10'] = big_clone_pre_10
x_rna_counts_lfc_clone_pre_10['small_clone_pre_10'] = small_clone_pre_10
x_rna_counts_lfc_clone_pre_10['clone_pre_10'] = np.where(big_clone_pre_10, 'big', np.where(small_clone_pre_10, 'small', 'NA'))
assert (x_rna_counts_lfc_clone_pre_10['clone_pre_10'] == 'big').sum() == (x_rna_counts_lfc_clone_pre_10['big_clone_pre_10']).sum(), "Assertion failed for big clone cells"
assert (x_rna_counts_lfc_clone_pre_10['clone_pre_10'] == 'small').sum() == (x_rna_counts_lfc_clone_pre_10['small_clone_pre_10']).sum(), "Assertion failed for small clone cells"
x_rna_counts_lfc_clone_pre_10 = x_rna_counts_lfc_clone_pre_10[x_rna_counts_lfc_clone_pre_10['clone_pre_10'] != 'NA']
print('x_rna_counts_lfc_clone_pre_10.shape', x_rna_counts_lfc_clone_pre_10.shape) # 5710, 19259
x_rna_counts_lfc_clone_pre_10.to_parquet(f"{data_folder_path}/seurat/data/x_rna_counts_lfc_clone_pre_10.parquet", engine='pyarrow')

## Export raw gene counts data (lfc_clone_pre_1 = big_clone_pre_1 - small_clone_pre_1)
x_rna_counts_lfc_clone_pre_1 = x_rna_counts.copy()
x_rna_counts_lfc_clone_pre_1 = x_rna_counts_lfc_clone_pre_1.iloc[:, genes_filters]
x_rna_counts_lfc_clone_pre_1['big_clone_pre_1'] = big_clone_pre_1
x_rna_counts_lfc_clone_pre_1['small_clone_pre_1'] = small_clone_pre_1
x_rna_counts_lfc_clone_pre_1['clone_pre_1'] = np.where(big_clone_pre_1, 'big', np.where(small_clone_pre_1, 'small', 'NA'))
assert (x_rna_counts_lfc_clone_pre_1['clone_pre_1'] == 'big').sum() == (x_rna_counts_lfc_clone_pre_1['big_clone_pre_1']).sum(), "Assertion failed for big clone cells"
assert (x_rna_counts_lfc_clone_pre_1['clone_pre_1'] == 'small').sum() == (x_rna_counts_lfc_clone_pre_1['small_clone_pre_1']).sum(), "Assertion failed for small clone cells"
x_rna_counts_lfc_clone_pre_1 = x_rna_counts_lfc_clone_pre_1[x_rna_counts_lfc_clone_pre_1['clone_pre_1'] != 'NA']
print('x_rna_counts_lfc_clone_pre_1.shape', x_rna_counts_lfc_clone_pre_1.shape) # 5710, 19259
x_rna_counts_lfc_clone_pre_1.to_parquet(f"{data_folder_path}/seurat/data/x_rna_counts_lfc_clone_pre_1.parquet", engine='pyarrow')

## Export raw gene counts data (lfc_expanded = post_expanded - pre_expanded)
# x_rna_counts_lfc_expanded = x_rna_counts.copy()
# x_rna_counts_lfc_expanded = x_rna_counts_lfc_expanded.iloc[:, genes_filters]
# x_rna_counts_lfc_expanded['blood_expanded_post'] = blood_expanded_post
# x_rna_counts_lfc_expanded['blood_expanded_pre'] = blood_expanded_pre
# x_rna_counts_lfc_expanded['expanded'] = np.where(blood_expanded_post, 'post', np.where(blood_expanded_pre, 'pre', 'NA'))
# assert (x_rna_counts_lfc_expanded['expanded'] == 'post').sum() == (x_rna_counts_lfc_expanded['blood_expanded_post']).sum(), "Assertion failed for post cells"
# assert (x_rna_counts_lfc_expanded['expanded'] == 'pre').sum() == (x_rna_counts_lfc_expanded['blood_expanded_pre']).sum(), "Assertion failed for pre cells"
# x_rna_counts_lfc_expanded = x_rna_counts_lfc_expanded[x_rna_counts_lfc_expanded['expanded'] != 'NA']
# print('x_rna_counts_lfc_expanded.shape', x_rna_counts_lfc_expanded.shape)
# x_rna_counts_lfc_expanded.to_parquet(f"{data_folder_path}/seurat/data/x_rna_counts_lfc_expanded.parquet", engine='pyarrow')

# ## Export raw gene counts data (lfc_nonexpanded = post_nonexpanded - pre_nonexpanded)
# x_rna_counts_lfc_nonexpanded = x_rna_counts.copy()
# x_rna_counts_lfc_nonexpanded = x_rna_counts_lfc_nonexpanded.iloc[:, genes_filters]
# x_rna_counts_lfc_nonexpanded['blood_nonexpanded_post'] = blood_nonexpanded_post
# x_rna_counts_lfc_nonexpanded['blood_nonexpanded_pre'] = blood_nonexpanded_pre
# x_rna_counts_lfc_nonexpanded['nonexpanded'] = np.where(blood_nonexpanded_post, 'post', np.where(blood_nonexpanded_pre, 'pre', 'NA'))
# assert (x_rna_counts_lfc_nonexpanded['nonexpanded'] == 'post').sum() == (x_rna_counts_lfc_nonexpanded['blood_nonexpanded_post']).sum(), "Assertion failed for post cells"
# assert (x_rna_counts_lfc_nonexpanded['nonexpanded'] == 'pre').sum() == (x_rna_counts_lfc_nonexpanded['blood_nonexpanded_pre']).sum(), "Assertion failed for pre cells"
# x_rna_counts_lfc_nonexpanded = x_rna_counts_lfc_nonexpanded[x_rna_counts_lfc_nonexpanded['nonexpanded'] != 'NA']
# print('x_rna_counts_lfc_nonexpanded.shape', x_rna_counts_lfc_nonexpanded.shape)
# x_rna_counts_lfc_nonexpanded.to_parquet(f"{data_folder_path}/seurat/data/x_rna_counts_lfc_nonexpanded.parquet", engine='pyarrow')

print('Done exporting data')

###############
############### Calculate the log fold change manually
###############

salient_genes = out_grads.mean(axis=0)
salient_genes = salient_genes

### First, export the data so that we can run FindAllMarkers in R ------------------------------------------------

### here, we calculated mean gene expressions for cells that satisfy the conditions
post_expanded_mean = x_rna[blood_expanded_post == 1].mean(axis=0)

# cross-check with saved data
adata_check = sc.AnnData(x_rna_counts_lfc_post.drop(columns=['blood_expanded_post', 'blood_nonexpanded_post', 'blood_post']))
sc.pp.normalize_total(adata_check, target_sum=1e4)
x_rna_counts_lfc_post_check = adata_check.X
post_expanded_mean_check = x_rna_counts_lfc_post_check[x_rna_counts_lfc_post['blood_post'] == 'expanded'].mean(axis=0)
assert (post_expanded_mean[genes_filters] - post_expanded_mean_check).sum() < 1e-6, "Assertion failed for post_expanded_mean"

pre_expanding_mean = x_rna[blood_expanding_pre == 1].mean(axis=0)
post_nonexpanded_mean = x_rna[blood_nonexpanded_post == 1].mean(axis=0)
pre_nonexpanding_mean = x_rna[blood_nonexpanding_pre == 1].mean(axis=0)
big_clone_pre_10_mean = x_rna[big_clone_pre_10 == 1].mean(axis=0)
small_clone_pre_10_mean = x_rna[small_clone_pre_10 == 1].mean(axis=0)
big_clone_pre_1_mean = x_rna[big_clone_pre_1 == 1].mean(axis=0)
small_clone_pre_1_mean = x_rna[small_clone_pre_1 == 1].mean(axis=0)

### calculate log fold changes 
pseudo_count = 1
# lfc of expanded_post - nonexpanded_post
lfc_post = np.log2(post_expanded_mean + pseudo_count) -  \
    np.log2(post_nonexpanded_mean + pseudo_count)
# lfc of expanded_post - expanded_pre
# lfc_expanded = np.log2(post_expanded_mean + pseudo_count) -  \
#     np.log2(pre_expanded_mean + pseudo_count)
# lfc of expanded_pre - nonexpanded_pre
lfc_pre = np.log2(pre_expanding_mean + pseudo_count) -  \
    np.log2(pre_nonexpanding_mean + pseudo_count)
# lfc of big_clone_pre_10 - small_clone_pre_10
lfc_clone_pre_10 = np.log2(big_clone_pre_10_mean + pseudo_count) -  \
    np.log2(small_clone_pre_10_mean + pseudo_count)
# lfc of big_clone_pre_1 - small_clone_pre_1
lfc_clone_pre_1 = np.log2(big_clone_pre_1_mean + pseudo_count) -  \
    np.log2(small_clone_pre_1_mean + pseudo_count)

## Gather lfc and gradients together ------------------------------------------------
diffexp_df_manual = pd.DataFrame(lfc_post[genes_filters], index=combined_data_columns[genes_filters], columns=['lfc_post'])
diffexp_df_manual['lfc_pre'] = lfc_pre[genes_filters]
# diffexp_df_manual['lfc_expanded'] = lfc_expanded[genes_filters]
diffexp_df_manual['lfc_clone_pre_10'] = lfc_clone_pre_10[genes_filters]
diffexp_df_manual['lfc_clone_pre_1'] = lfc_clone_pre_1[genes_filters]
diffexp_df_manual['salient_genes'] = salient_genes[genes_filters]

diffexp_df_manual['post_expanded_mean'] = post_expanded_mean[genes_filters]
diffexp_df_manual['post_nonexpanded_mean'] = post_nonexpanded_mean[genes_filters]
diffexp_df_manual['pre_expanding_mean'] = pre_expanding_mean[genes_filters]
diffexp_df_manual['pre_nonexpanding_mean'] = pre_nonexpanding_mean[genes_filters]
diffexp_df_manual['big_clone_pre_10_mean'] = big_clone_pre_10_mean[genes_filters]
diffexp_df_manual['small_clone_pre_10_mean'] = small_clone_pre_10_mean[genes_filters]
diffexp_df_manual['big_clone_pre_1_mean'] = big_clone_pre_1_mean[genes_filters]
diffexp_df_manual['small_clone_pre_1_mean'] = small_clone_pre_1_mean[genes_filters]

# check for any inf/-inf/nan values in logfoldchange_pre and logfoldchange_post
assert(np.isinf(diffexp_df_manual['lfc_post']).sum() == np.isinf(diffexp_df_manual['lfc_pre']).sum() == 0)
assert(np.isnan(diffexp_df_manual['lfc_clone_pre_10']).sum() == np.isnan(diffexp_df_manual['lfc_clone_pre_1']).sum() == 0)

# save diffexp_df_manual
diffexp_df_manual.to_csv(f'{data_folder_path}/diffexp_df_manual.csv', index=True)

###############
############### Calculate the log fold change using scanpy
###############

### Conduct DE analysis using scanpy ------------------------------------------------
adata_rna.obs['blood_expanded_post'] = blood_expanded_post
adata_rna.obs['blood_nonexpanded_post'] = blood_nonexpanded_post
adata_rna.obs['blood_post'] = np.where(blood_expanded_post, 'expanded', np.where(blood_nonexpanded_post, 'non-expanded', 'NA'))
assert (adata_rna.obs['blood_post'] == 'expanded').sum() == (adata_rna.obs['blood_expanded_post']).sum(), "Assertion failed for expanded cells"
assert (adata_rna.obs['blood_post'] == 'non-expanded').sum() == (adata_rna.obs['blood_nonexpanded_post']).sum(), "Assertion failed for non-expanded cells"

adata_rna.obs['blood_expanding_pre'] = blood_expanding_pre
adata_rna.obs['blood_nonexpanding_pre'] = blood_nonexpanding_pre
adata_rna.obs['blood_pre'] = np.where(blood_expanding_pre, 'expanding', np.where(blood_nonexpanding_pre, 'non-expanding', 'NA'))
assert (adata_rna.obs['blood_pre'] == 'expanding').sum() == (adata_rna.obs['blood_expanding_pre']).sum(), "Assertion failed for expanded cells"
assert (adata_rna.obs['blood_pre'] == 'non-expanding').sum() == (adata_rna.obs['blood_nonexpanding_pre']).sum(), "Assertion failed for non-expanded cells"

adata_rna.obs['big_clone_pre_10'] = big_clone_pre_10
adata_rna.obs['small_clone_pre_10'] = small_clone_pre_10
adata_rna.obs['clone_pre_10'] = np.where(big_clone_pre_10, 'big', np.where(small_clone_pre_10, 'small', 'NA'))
assert (adata_rna.obs['clone_pre_10'] == 'big').sum() == (adata_rna.obs['big_clone_pre_10']).sum(), "Assertion failed for big clone cells"
assert (adata_rna.obs['clone_pre_10'] == 'small').sum() == (adata_rna.obs['small_clone_pre_10']).sum(), "Assertion failed for small clone cells"

adata_rna.obs['big_clone_pre_1'] = big_clone_pre_1
adata_rna.obs['small_clone_pre_1'] = small_clone_pre_1
adata_rna.obs['clone_pre_1'] = np.where(big_clone_pre_1, 'big', np.where(small_clone_pre_1, 'small', 'NA'))
assert (adata_rna.obs['clone_pre_1'] == 'big').sum() == (adata_rna.obs['big_clone_pre_1']).sum(), "Assertion failed for big clone cells"
assert (adata_rna.obs['clone_pre_1'] == 'small').sum() == (adata_rna.obs['small_clone_pre_1']).sum(), "Assertion failed for small clone cells"

# # Filter genes expressed in more than 10 cells
sc.pp.filter_genes(adata_rna, min_cells=11) 
print(adata_rna.shape) # 73961, 19256
# sc.tl.rank_genes_groups(adata_rna, 'blood_post', reference="non-expanded", method='wilcoxon')
# sc.tl.rank_genes_groups(adata_rna, 'clone_pre_10', reference="small", method='wilcoxon')
# sc.tl.rank_genes_groups(adata_rna, 'clone_pre_1', reference="small", method='wilcoxon')

# Perform rank_genes_groups for different comparisons
comparisons = [
    ('blood_post', 'expanded', 'non-expanded', 'lfc_post'), # group, reference
    ('clone_pre_10', 'big', 'small', 'lfc_clone_pre_10'),
    ('clone_pre_1', 'big', 'small', 'lfc_clone_pre_1'),
    ('blood_pre', 'expanding', 'non-expanding', 'lfc_pre'),
]

diffexp_df_sc = pd.DataFrame(index=adata_rna.var_names)
for group, compare, reference, lfc_name in comparisons:
    print(f"Running rank_genes_groups for group: {group} with compare: {compare} and ref: {reference}")
    adata_rna_temp = adata_rna[adata_rna.obs[group] != 'NA'].copy()
    sc.tl.rank_genes_groups(adata_rna_temp, group, reference=reference, method='wilcoxon', rankby_abs=True)
    
    # Extract results
    results_temp = adata_rna_temp.uns['rank_genes_groups']
    for metric in ['logfoldchanges', 'pvals', 'pvals_adj']:
        col_name = f"{lfc_name}_{metric}"
        diffexp_df_sc[col_name] = pd.Series(
            results_temp[metric][compare],
            index=results_temp['names'][compare]
        )
    
    # Rename column to remove logfoldchanges
    diffexp_df_sc.rename(columns={f"{lfc_name}_logfoldchanges": lfc_name}, inplace=True)

diffexp_df_sc['salient_genes'] = salient_genes[genes_filters]
# save diffexp_df_sc
diffexp_df_sc.to_csv(f'{data_folder_path}/diffexp_df_sc.csv', index=True)
