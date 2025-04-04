import os
import numpy as np
import random
random.seed(0)
np.random.seed(0)

os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=32
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

load_precalculated_umap_real = True # set to True if you want to load the precalculated UMAP for real data
skip_3e = False
skip_3g = False
skip_supp_fig_4b = False

import pandas as pd
import scipy.io
import sklearn.decomposition
import sklearn.svm
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import umap
import sys
import time
import termcolor
import sklearn.cluster
import logomaker as lm
import scanpy
import math
import glob
import json
import argparse
import statsmodels.api as sm
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from helpers import *
fig = plt.figure()
old_fig_size = fig.get_size_inches()

#############################################
#############################################
# load args
output_folder = 'holdout1'
output_folder = os.path.join('/data/che/TRIM/HNSCC/output', output_folder)
args = argparse.ArgumentParser().parse_args()
with open(os.path.join(output_folder, 'args.txt'), 'r') as f:
    args.__dict__ = json.load(f)

# CH: update processed_data_folder to read from my folder
args.processed_data_folder = '/home/che/TRIM/data_processed'

#############################################
#############################################
# load data
print('Starting to load data...')
t = time.time()

with open(os.path.join(args.processed_data_folder, 'data_rna.npz'), 'rb') as f:
    npzfile = np.load(f)
    data_rna_raw = npzfile['data_rna']

with open(os.path.join(args.processed_data_folder, 'data_tcr.npz'), 'rb') as f:
    npzfile = np.load(f)
    data_tcr = npzfile['data_tcr']

with open(os.path.join(args.processed_data_folder, 'data_labels.npz'), 'rb') as f:
    npzfile = np.load(f)
    data_labels = npzfile['data_labels']

with open(os.path.join(args.processed_data_folder, 'data_all_tcrs.npz'), 'rb') as f:
    npzfile = np.load(f, allow_pickle=True)
    df_all_tcrs = npzfile['data_all_tcrs']
    rows = npzfile['rows']
    cols = npzfile['cols']
    df_all_tcrs = pd.DataFrame(df_all_tcrs, index=rows, columns=cols)

with open(os.path.join(args.processed_data_folder, 'combined_data_columns.npz'), 'rb') as f:
    npzfile = np.load(f, allow_pickle=True)
    combined_data_columns = npzfile['cols']

col_bloodtumor, col_prepost, col_celltype, col_patient, col_tcr, col_tcr_v, col_tcr_j, col_treatment = list(range(data_labels.shape[1]))
ID_NULL_TCR = 0
df_blood_metadata = pd.read_csv('/home/che/TRIM/data/BloodCD4_metadata.csv', header=0, index_col=0)
patient_ids = ['P23', 'P24', 'P29', 'P32', 'P01', 'P02', 'P04', 'P05', 'P08', 'P09', 'P10', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P25', 'P26', 'P28',  'P30', 'P31']
print('Loaded data in {:.1f} s'.format(time.time() - t))

#############################################
#############################################
# preprocess data
print('Starting to pca/viz_umap data...')
t = time.time()

args.umap_trained_file = 'umap_trained.pkl'

# Currently, args.reduction = 'PCA'
if args.reduction == 'PCA':
    npca = 100
    pca = sklearn.decomposition.PCA(npca, random_state=0)
    data_rna = pca.fit_transform(data_rna_raw)

elif args.reduction == 'HVG':
    nhvg = 1000
    mask_hvg = scanpy.pp.highly_variable_genes(scanpy.AnnData(data_rna, pd.DataFrame(data_rna).index.to_frame(), pd.DataFrame(data_rna).columns.to_frame()), inplace=False, n_top_genes=nhvg)['highly_variable'].values
    data_rna = data_rna[:, mask_hvg]

if not os.path.exists(args.umap_trained_file):
    viz_reducer = umap.UMAP(random_state=0).fit(data_rna)
    pickle.dump(viz_reducer, open(args.umap_trained_file, 'wb+'))
    print('Saved umap_trained.pkl to', args.umap_trained_file)
else:
    viz_reducer = pickle.load(open(args.umap_trained_file, 'rb'))
    print('Loaded umap_trained.pkl from', args.umap_trained_file)

# save e_eval_reals for reproducibility as UMAP.transform is stochastic
args.e_eval_reals_path = '/home/che/TRIM/data_figures/HNSCC/e_eval_reals.pickle'
if os.path.exists(args.e_eval_reals_path):
    if load_precalculated_umap_real:
        with open(args.e_eval_reals_path, 'rb') as f:
            e_eval_reals = pickle.load(f)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!! Loaded e_eval_reals.pickle from', args.e_eval_reals_path)
        print('!!!!!!! if the processed data is updated, please delete the file and re-run to generate a new e_eval_reals.pickle')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        raise Exception('Found e_eval_reals.pickle but load_precalculated_umap_real is set to False. Please delete the file or set load_precalculated_umap_real to True to load the existing file.')
else:
    e_eval_reals = viz_reducer.transform(data_rna)
    with open(args.e_eval_reals_path, 'wb') as f:
        pickle.dump(e_eval_reals, f)
    print('Saved e_eval_reals.pickle to', args.e_eval_reals_path)
print('Finished pca/viz_umap data in {:.1f} s'.format(time.time() - t))

#############################################
#############################################
# load model
print('Starting to load model...')
t = time.time()
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

x_rna = data_rna
x_tcr = data_tcr
x_label = data_labels

mask_preprocess = x_label[:, col_tcr] != ID_NULL_TCR
x_rna = x_rna[mask_preprocess]
x_tcr = x_tcr[mask_preprocess]
x_label = x_label[mask_preprocess]
e_eval_reals = e_eval_reals[mask_preprocess]

G = Generator(args)
G.load_state_dict(torch.load(os.path.join(output_folder, 'model.pth')))

G.eval()
G = G.to(device)

print('Loaded model in {:.1f} s'.format(time.time() - t))

#############################################
#############################################
# load output
print('Starting to load output...')
t = time.time()

with open(os.path.join(output_folder, 'preds.npz'), 'rb') as f:
    npzfile = np.load(f)
    preds_rna = npzfile['preds_rna']
    preds_tcr = npzfile['preds_tcr']
    recon_rna_z = npzfile['recon_rna_z']
    recon_tcr_z = npzfile['recon_tcr_z']

    pseudo_tcrs = npzfile['pseudo_tcrs']
    tcr_dists = npzfile['tcr_dists']
    thresh_fitted = npzfile['thresh_fitted']


print('Loaded output in {:.1f} s'.format(time.time() - t))

#############################################
#############################################

print('Starting to load holdout output...')
t = time.time()
preds_rna_holdout = np.zeros_like(preds_rna)
pseudo_tcrs_holdout = np.zeros_like(pseudo_tcrs)
df = pd.DataFrame()
for i_pid in range(27):
    print('Patient {}'.format(i_pid))
    fn = '/data/che/TRIM/HNSCC/output/holdout{}'.format(i_pid)
    args = argparse.ArgumentParser().parse_args()
    with open(os.path.join(fn, 'args.txt'), 'r') as f:
        args.__dict__ = json.load(f)

    with open(os.path.join(fn, 'preds.npz'), 'rb') as f:
        npzfile = np.load(f)
        preds_rna = npzfile['preds_rna']
        pseudo_tcrs = npzfile['pseudo_tcrs']
    mask = x_label[:, col_patient] == i_pid
    preds_rna_holdout = np.where(mask[:, np.newaxis], preds_rna, preds_rna_holdout)

    offset = pseudo_tcrs_holdout.max()
    dict_old2new = {tcr: i + offset for i, tcr in enumerate(np.unique(pseudo_tcrs[mask]))}

    def myfunc(tmp):
        if tmp[0] in dict_old2new:
            return dict_old2new[tmp[0]]
        else:
            return -1

    new_tcr = np.apply_along_axis(myfunc, 1, pseudo_tcrs[:, np.newaxis])
    pseudo_tcrs_holdout = np.where(mask, new_tcr, pseudo_tcrs_holdout)

print('Loaded output in {:.1f} s'.format(time.time() - t))

#############################################
#############################################


#########################
# for all ambient-rna gene panels
data_rna_ambient = pca.inverse_transform(data_rna[mask_preprocess])
preds_rna_ambient = pca.inverse_transform(preds_rna_holdout)
classifier_cd8 = sklearn.svm.SVC()
classifier_cd8.fit(x_rna, x_label[:, col_celltype])

gene_col_dict = {
    'GZMB': 15404,
    'GZMK': 15406,
    'MKI67': 19901,
    'KLF2': 17193,
    'TCF7': 26577, # [26577, 26578, 26579, 26580],
    'ITGAE': 16757,
    'CTLA4': 12257,
    'PDCD1': 21953,
    'ZNF683': 29544,
    }

# for i in range(combined_data_columns.shape[0]):
#     col = combined_data_columns[i]
#     if 'znf' in col.lower():
#         print(i, col)

#############################################
# up to here, loads data and a specific trained model
#######################################################################################################################
# down from here, any EVALUATION...
#############################################


#############################################
#############################################
# FIGURES
figure_path = '/home/che/TRIM/git/tcr/figures'
#########################

def make_legend(ax, labels, colors, center=[0, 0], loc='upper right', ncol=1, s=100):
    numlabs = len(labels)
    for i, label in enumerate(labels):
        ax.scatter(center[0], center[1], s=s, c=colors[i], label=label)
    ax.legend(loc=loc, ncol=ncol)

# figure 2a
fig.clf()
ax = fig.subplots(1, 1)
# make_legend(ax, ['Blood', 'Tumor'], cmap=mpl.cm.viridis, center=[5, 5], loc='lower left', ncol=2)
rng = np.random.RandomState(0)
r = rng.choice(range(e_eval_reals.shape[0]), e_eval_reals.shape[0], replace=False)
colors = np.where(x_label[r, col_bloodtumor] == 0, '#E41A1C', '#377EB8')
# ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], c=x_label[r, col_bloodtumor], s=1, cmap=mpl.cm.viridis)
ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], c=colors, s=1)
ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_bloodtumor.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_bloodtumor.svg', format='svg')

fig.clf()
fig, ax = plt.subplots(1, 1)
colors = ['#E41A1C', '#377EB8']
make_legend(ax, ['Blood', 'Tumor'], colors)
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_bloodtumor_legend.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_bloodtumor_legend.svg', format='svg')

fig.clf()
ax = fig.subplots(1, 1)
# make_legend(ax, ['CD4', 'CD8'], cmap=mpl.cm.viridis, center=[5, 5], loc='lower left', ncol=2)
r = rng.choice(range(e_eval_reals.shape[0]), e_eval_reals.shape[0], replace=False)
colors = np.where(x_label[r, col_celltype] == 0, '#F781BF', '#999999')
# ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], c=x_label[r, col_celltype], s=1, cmap=mpl.cm.viridis)
ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], c=colors, s=1)
ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_cd4cd8.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_cd4cd8.svg', format='svg')

fig.clf()
fig, ax = plt.subplots(1, 1)
colors = ['#F781BF', '#999999']
make_legend(ax, ['CD4', 'CD8'], colors)
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_cd4cd8_legend.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_cd4cd8_legend.svg', format='svg')

fig.clf()
ax = fig.subplots(1, 1)
# make_legend(ax, ['Pre-treatment', 'Post-treatment'], cmap=mpl.cm.viridis, center=[5, 5], loc='lower left', ncol=2)
np.random.seed(0)
N = 3500
rs = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            mask = x_label[:, col_prepost] == i
            mask = np.logical_and(mask, x_label[:, col_bloodtumor] == j)
            mask = np.logical_and(mask, x_label[:, col_celltype] == k)
            r_ = np.random.choice(np.argwhere(mask).reshape(-1), N, replace=False)
            rs.append(r_)
rs = np.concatenate(rs)
r = np.random.choice(rs, rs.shape[0], replace=False)
colors = np.where(x_label[r, col_prepost] == 0, '#4DAF4A', '#984EA3')
# ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], c=x_label[r, col_prepost], s=1, cmap=mpl.cm.viridis)
ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], c=colors, s=1)
ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_prepost.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_prepost.svg', format='svg')

fig.clf()
fig, ax = plt.subplots(1, 1)
colors = ['#4DAF4A', '#984EA3']
make_legend(ax, ['Pre-treatment', 'Post-treatment'], colors)
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_prepost_legend.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_prepost_legend.svg', format='svg')

fig.clf()
ax = fig.subplots(1, 1)
np.random.seed(0)
c = np.take(df_all_tcrs.fillna(0).sum(axis=1).values, x_label[:, col_tcr].astype(np.int32))
c = c / c.sum() * 1000000
r = np.random.choice(range(e_eval_reals.shape[0]), e_eval_reals.shape[0], replace=False)
ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], s=1, cmap=mpl.cm.viridis,  c=np.log(c[r]), vmin=0, vmax=2.3) # np.log(10/c.sum()*1000000) ~= 2.3
[ax.set_xticks([]), ax.set_yticks([])]
[ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_real_clonecount.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_real_clonecount.svg', format='svg')

fig.clf()
ax = fig.subplots(1, 1)
np.random.seed(0)
# make_legend2(ax, ['P{:.0f}'.format(i) for i in sorted(np.unique(x_label[:, col_patient]))], cmap=mpl.cm.tab20, ncol=3)
# make_legend2(ax, ['P{:.0f}'.format(i) for i in sorted(np.unique(x_label[:, col_patient]))], cmap=mpl.cm.tab20, ncol=2)
r = np.random.choice(range(e_eval_reals.shape[0]), e_eval_reals.shape[0], replace=False)
ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], s=1, cmap=mpl.cm.tab20,  c=x_label[r, col_patient])
[ax.set_xticks([]), ax.set_yticks([])]
[ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_patient_labels.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_patient_labels.svg', format='svg')

fig.clf()
ax = fig.subplots(1, 1)
make_legend2(ax, ['P{:.0f}'.format(i) for i in sorted(np.unique(x_label[:, col_patient]))], cmap=mpl.cm.tab20, ncol=2)
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_patient_labels_legend.png')
fig.savefig(f'{figure_path}/figure3/3a_rna_umap_by_patient_labels_legend.svg', format='svg')

#########################
# figure 3b
# rna pairwise dist vs hamming pairwise dist
# skip this, for this analysis, see /home/che/TRIM/git/tcr/analysis/HNSCC/rna_pairwise_dist.py

#########################
# figure 3c
# tcr embeddings pairwise distances vs rna
np.random.seed(0)
tmp1 = []
tmp2 = []
for i in np.unique(x_label[:, col_tcr]):
    tmp1.append(x_tcr[x_label[:, col_tcr] == i].mean(axis=0)[np.newaxis, :])
    # tmp1.append(tcrbert_embeddings[int(i)][np.newaxis, :])
    tmp2.append(x_rna[x_label[:, col_tcr] == i].mean(axis=0)[np.newaxis, :])
    
tmp1 = np.array(tmp1).squeeze(1)
tmp2 = np.array(tmp2).squeeze(1)

mask = np.random.choice(range(tmp1.shape[0]), 20000, replace=False)
dists1 = sklearn.metrics.pairwise_distances(tmp1[mask], tmp1[mask])
dists2 = sklearn.metrics.pairwise_distances(tmp2[mask], tmp2[mask])
mask_triu = (np.triu(np.ones([dists1.shape[0], dists1.shape[0]])) - np.eye(dists1.shape[0])).flatten() == 1
r = np.corrcoef(dists1.flatten()[mask_triu], dists2.flatten()[mask_triu])
print(r)
fig.clf()
ax = fig.subplots(1, 1)
mask_triu = np.logical_and(np.random.uniform(0, 1, mask_triu.shape) < .001, mask_triu)
ax.scatter(dists1.flatten()[mask_triu], dists2.flatten()[mask_triu], s=1, alpha=.5)
# ax.set_xlabel('TCR pairwise distance')
# ax.set_ylabel('Gene expression pairwise distance')

model_results = sm.OLS(dists2.flatten()[mask_triu], sm.add_constant(dists1.flatten()[mask_triu])).fit()
xlim = ax.get_xlim()
ax.plot(np.arange(xlim[0], xlim[1], .01), model_results.params[0] + model_results.params[1] * np.arange(xlim[0], xlim[1], .01), c='k', linestyle='--', label='r = {:0.3f}'.format(np.sign( model_results.params[1]) * np.sqrt(model_results.rsquared)))
ax.legend(loc='lower right')
fig.savefig(f'{figure_path}/figure3/3c_pairwise_distances_rna_by_tcr.png')
fig.savefig(f'{figure_path}/figure3/3c_pairwise_distances_rna_by_tcr.svg', format='svg')

# figure 3c 
# tcr bert embeddings pairwise distances vs rna
# load in tcr bert embeddings in pickle file
with open('/home/che/TRIM/data_figures/HNSCC/tcrbert_embeddings.pkl', 'rb') as f:
    tcrbert_embeddings = pickle.load(f)

np.random.seed(0)
tmp1 = []
tmp2 = []
for i in np.unique(x_label[:, col_tcr]):
    tmp1.append(tcrbert_embeddings[int(i)][np.newaxis, :])
    tmp2.append(x_rna[x_label[:, col_tcr] == i].mean(axis=0)[np.newaxis, :])
    
tmp1 = np.array(tmp1).squeeze(1)
tmp2 = np.array(tmp2).squeeze(1)

mask = np.random.choice(range(tmp1.shape[0]), 20000, replace=False)
dists1 = sklearn.metrics.pairwise_distances(tmp1[mask], tmp1[mask])
dists2 = sklearn.metrics.pairwise_distances(tmp2[mask], tmp2[mask])
mask_triu = (np.triu(np.ones([dists1.shape[0], dists1.shape[0]])) - np.eye(dists1.shape[0])).flatten() == 1
r = np.corrcoef(dists1.flatten()[mask_triu], dists2.flatten()[mask_triu])
print(r)
fig.clf()
ax = fig.subplots(1, 1)
mask_triu = np.logical_and(np.random.uniform(0, 1, mask_triu.shape) < .001, mask_triu)
ax.scatter(dists1.flatten()[mask_triu], dists2.flatten()[mask_triu], s=1, alpha=.5)
# ax.set_xlabel('TCR pairwise distance')
# ax.set_ylabel('Gene expression pairwise distance')

model_results = sm.OLS(dists2.flatten()[mask_triu], sm.add_constant(dists1.flatten()[mask_triu])).fit()
xlim = ax.get_xlim()
ax.plot(np.arange(xlim[0], xlim[1], .01), model_results.params[0] + model_results.params[1] * np.arange(xlim[0], xlim[1], .01), c='k', linestyle='--', label='r = {:0.3f}'.format(np.sign( model_results.params[1]) * np.sqrt(model_results.rsquared)))
ax.legend(loc='lower right')
fig.savefig(f'{figure_path}/figure3/3c_tcrbert_pairwise_distances_rna_by_tcr.png')
fig.savefig(f'{figure_path}/figure3/3c_tcrbert_pairwise_distances_rna_by_tcr.svg', format='svg')

#########################
# figure 3d
# make low-clonality plots
high_clone = False
if high_clone:
    mask = np.take(df_all_tcrs.fillna(0).sum(axis=1).values <= 10, x_label[:, col_tcr].astype(int), axis=0) # plot umap for counts > 10
else:
    mask = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0) # plot umap for counts <= 10
c = np.take(df_all_tcrs.fillna(0).sum(axis=1).values, x_label[:, col_tcr].astype(int), axis=0)
c = c / c.sum() * 1000000 # apply the same normalization as above 
log_c = np.log(c)

fig.clf()
ax = fig.subplots(1, 1)
if high_clone:
    sc = ax.scatter(e_eval_reals[~mask, 0], e_eval_reals[~mask, 1], s=1, c=log_c[~mask], cmap=mpl.cm.winter, vmin=2.3) #color map for counts > 10, plot log of counts
    # sc = ax.scatter(e_eval_reals[~mask, 0], e_eval_reals[~mask, 1], s=1, c=c[~mask], cmap=mpl.cm.winter) # color map for counts > 10
else:
    # ax.scatter(e_eval_reals[mask, 0], e_eval_reals[mask, 1], s=1, c=data_rna_ambient[mask, 10976], cmap=mpl.cm.bwr)
    sc = ax.scatter(e_eval_reals[~mask, 0], e_eval_reals[~mask, 1], s=1, c=np.log(c[~mask]), vmin=0, vmax=2.3)
[ax.set_xticks([]), ax.set_yticks([])]
[ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
umap_lims = [ax.get_xlim(), ax.get_ylim()]
[ax.set_xlim(umap_lims[0]), ax.set_ylim(umap_lims[1])]
# Add color bar
if high_clone:
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Log(Clone Count)')
    cbar.ax.tick_params(size=0)
else:
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Log(Clone Count)')
    cbar.ax.tick_params(size=0)
# cbar.set_ticks([10, 500])
fig.savefig(f"{figure_path}/figure3/3d_{'high' if high_clone else 'low'}_colonality_log.png")
fig.savefig(f"{figure_path}/figure3/3d_{'high' if high_clone else 'low'}_colonality_log.svg", format='svg')


# make high-clonality plots
high_clone = True
if high_clone:
    mask = np.take(df_all_tcrs.fillna(0).sum(axis=1).values <= 10, x_label[:, col_tcr].astype(int), axis=0) # plot umap for counts > 10
else:
    mask = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0) # plot umap for counts <= 10
c = np.take(df_all_tcrs.fillna(0).sum(axis=1).values, x_label[:, col_tcr].astype(int), axis=0)
c = c / c.sum() * 1000000 # apply the same normalization as above 
log_c = np.log(c)

fig.clf()
ax = fig.subplots(1, 1)
if high_clone:
    sc = ax.scatter(e_eval_reals[~mask, 0], e_eval_reals[~mask, 1], s=1, c=log_c[~mask], cmap=mpl.cm.winter, vmin=2.3) #color map for counts > 10, plot log of counts
    # sc = ax.scatter(e_eval_reals[~mask, 0], e_eval_reals[~mask, 1], s=1, c=c[~mask], cmap=mpl.cm.winter) # color map for counts > 10
else:
    # ax.scatter(e_eval_reals[mask, 0], e_eval_reals[mask, 1], s=1, c=data_rna_ambient[mask, 10976], cmap=mpl.cm.bwr)
    sc = ax.scatter(e_eval_reals[~mask, 0], e_eval_reals[~mask, 1], s=1, c=np.log(c[~mask]), vmin=0, vmax=2.3)
[ax.set_xticks([]), ax.set_yticks([])]
[ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
umap_lims = [ax.get_xlim(), ax.get_ylim()]
[ax.set_xlim(umap_lims[0]), ax.set_ylim(umap_lims[1])]
# Add color bar
if high_clone:
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Log(Clone Count)')
    cbar.ax.tick_params(size=0)
else:
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Log(Clone Count)')
    cbar.ax.tick_params(size=0)
# cbar.set_ticks([10, 500])
fig.savefig(f"{figure_path}/figure3/3d_{'high' if high_clone else 'low'}_colonality_log.png")
fig.savefig(f"{figure_path}/figure3/3d_{'high' if high_clone else 'low'}_colonality_log.svg", format='svg')

#########################
# figure 3e/3f

# cytotoxic t cells
# ninegene_sig = [17409, 14748, 10976, 22884, 15403, 15405, 16695, 20891, 23104]
# ninegene_sig_names = ['LAG3', 'GBP5', 'CCL5', 'PRF1', 'GZMA', 'GZMH', 'IRF1', 'NKG7', 'PSMB10']
# ninegene_sig = [17409, 22884, 15405, 23104]
# ninegene_sig_names = ['LAG3', 'PRF1', 'GZMH',  'PSMB10']

# r = np.random.choice(range(e_eval_reals.shape[0]), e_eval_reals.shape[0], replace=False)
# fig.set_size_inches([10, 2.5])
# fig.clf()
# axes = fig.subplots(3, 3)
# # offset = 100
# for i, ax in enumerate(axes.flatten()):
#     ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], s=1, cmap=mpl.cm.bwr, c=data_rna_ambient[r, ninegene_sig[i]])
#     [ax.set_xticks([]), ax.set_yticks([])]
#     [ax.set_xlim(umap_lims[0]), ax.set_ylim(umap_lims[1])]
#     ax.set_title(ninegene_sig_names[i])
# fig.savefig('figures/tmp.png')
# fig.set_size_inches(old_fig_size)


# mask = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0)
# mask1 = np.logical_and(mask, x_label[:, col_bloodtumor] == 1)
# mask1 = np.logical_and(mask1, e_eval_reals[:, 0] > 11)
# mask1 = np.logical_and(mask1, e_eval_reals[:, 1] > 3.5)
# mask1 = np.logical_and(mask1, x_label[:, col_celltype] == 1)

# mask2 = np.logical_and(mask, x_label[:, col_bloodtumor] == 1)
# mask2 = np.logical_and(mask2, np.logical_or(e_eval_reals[:, 0] < 11, e_eval_reals[:, 1] < 3.5))
# mask2 = np.logical_and(mask2, x_label[:, col_celltype] == 1)

# out = get_differentially_expressed_genes(pd.DataFrame(np.concatenate([data_rna_ambient[mask1], data_rna_ambient[mask2]], axis=0)), np.concatenate([np.zeros(mask1.sum()), np.ones(mask2.sum())], axis=0), .01)

if not skip_3e:
    fig.set_size_inches([5, 5])
    fig.clf()
    axes = fig.subplots(2, 2)
    ind = 0
    # get the indices of the 4 desired TCRs in the figure: CASSQVGEGTEAFF, CASSRGLAHQPQHF, CASSMTGGAYEQYF, CASSVVWGLSTEAFF
    search_tcrs = ['CASSQVGEGTEAFF', 'CASSRGLAHQPQHF', 'CASSMTGGAYEQYF', 'CASSVVWGLSTEAFF']
    search_tcrs = [tcr + ' ' * (80 - len(tcr)) for tcr in search_tcrs] # null pad to match length = 80
    index_list = [df_all_tcrs.index.to_list().index(search_tcrs[0]), df_all_tcrs.index.to_list().index(search_tcrs[1]), df_all_tcrs.index.to_list().index(search_tcrs[2]), df_all_tcrs.index.to_list().index(search_tcrs[3])]
    # index_list = 9, 21, 33, 42
    target_bloodtumor = 0 # blood = 0, tumor = 1
    target_celltype = 1 # cd4 = 0, cd8 = 1
    for i, ax in enumerate(axes.flatten()):
        # Note from Matt: while good loop code helped me search for "representative" TCRs that would make the point clearly, and then also plot them. 
        # Given the goal here is to reproduce the previous plot, I would suggest skipping the "while good" loop from lines 829-850, 
        # and instead get the ind of the 4 desired TCRs (the ones in the title of the 4 plots), create the mask with:
        # Skip from here -------------------
        # good = -1
        # while good != 1:
        #     ind += 2
        #     if not (x_label[:, col_tcr] == ind).sum() > 10:
        #         continue
        #     ax.set_title(df_all_tcrs.index[ind].replace(' ', ''), fontsize=6)
        #     mask = x_label[:, col_tcr] == ind
        #     vals, counts = np.unique(x_label[mask, col_bloodtumor], return_counts=True)
        #     good1 = vals[counts.argmax()] == target_bloodtumor
        #     if counts.max() / counts.sum() < .9: good1 = -1

        #     vals, counts = np.unique(x_label[mask, col_celltype], return_counts=True)
        #     good2 = vals[counts.argmax()] == target_bloodtumor
        #     if counts.max() / counts.sum() < .9: good2 = -1
            
        #     good = np.logical_and(good1, good2).astype(int)

        #     # if data_rna_ambient[mask, 100].mean() > np.percentile(data_rna_ambient[:, 100], 90): good = -1

        #     mask = np.logical_and(x_label[:, col_tcr] == ind, x_label[:, col_bloodtumor] == target_bloodtumor)
        #     mask = np.logical_and(mask, x_label[:, col_celltype] == target_celltype)
        #     if mask.sum() < 10: good = -1
        # Skip ends -------------------------------

        # Create mask for the 4 desired TCRs -------------------
        ind = index_list[i]
        ax.set_title(df_all_tcrs.index[ind].replace(' ', ''), fontsize=6)
        mask = np.logical_and(x_label[:, col_tcr] == ind, x_label[:, col_bloodtumor] == target_bloodtumor)
        mask = np.logical_and(mask, x_label[:, col_celltype] == target_celltype)
        # End of mask creation -------------------------------

        tmp = e_eval_reals[mask]
        ax.scatter(e_eval_reals[:, 0], e_eval_reals[:, 1], c='darkgray', s=1, alpha=.1)
        ax.scatter(tmp[:, 0], tmp[:, 1], s=1, c='b')
        [ax.set_xticks([]), ax.set_yticks([])]
        [ax.set_xlim(umap_lims[0]), ax.set_ylim(umap_lims[1])]
    fig.savefig(f'{figure_path}/figure3/3e_highlight_one_tcr_on_umap_cd8_blood.png')
    fig.savefig(f'{figure_path}/figure3/3e_highlight_one_tcr_on_umap_cd8_blood.svg', format='svg')
    fig.set_size_inches(old_fig_size)
else:
    print('Skipping figure 3e for faster evaluation...')

if not skip_3g:
    fig.set_size_inches([5, 5])
    fig.clf()
    axes = fig.subplots(2, 2)
    ind = 0
    # get the indices of the 4 desired TCRs in the figure
    search_tcrs = ['CASGIGTSGYNEQFF', 'CASREGLSYEQYF', 'CASSFVSGANVLTF', 'CAISANSGRNDTQYF']
    search_tcrs = [tcr + ' ' * (80 - len(tcr)) for tcr in search_tcrs] # null pad to match length = 80
    index_list = [df_all_tcrs.index.to_list().index(search_tcrs[0]), df_all_tcrs.index.to_list().index(search_tcrs[1]), df_all_tcrs.index.to_list().index(search_tcrs[2]), df_all_tcrs.index.to_list().index(search_tcrs[3])]
    # index_list = 9, 21, 33, 42
    target_bloodtumor = 1 # blood = 0, tumor = 1
    target_celltype = 1 # cd4 = 0, cd8 = 1
    for i, ax in enumerate(axes.flatten()):
        # Create mask for the 4 desired TCRs -------------------
        ind = index_list[i]
        ax.set_title(df_all_tcrs.index[ind].replace(' ', ''), fontsize=6)
        mask = np.logical_and(x_label[:, col_tcr] == ind, x_label[:, col_bloodtumor] == target_bloodtumor)
        mask = np.logical_and(mask, x_label[:, col_celltype] == target_celltype)
        # End of mask creation -------------------------------

        tmp = e_eval_reals[mask]
        ax.scatter(e_eval_reals[:, 0], e_eval_reals[:, 1], c='darkgray', s=1, alpha=.1)
        ax.scatter(tmp[:, 0], tmp[:, 1], s=1, c='b')
        [ax.set_xticks([]), ax.set_yticks([])]
        [ax.set_xlim(umap_lims[0]), ax.set_ylim(umap_lims[1])]
    fig.savefig(f'{figure_path}/figure3/3e_highlight_one_tcr_on_umap_cd8_tumor.png')
    fig.savefig(f'{figure_path}/figure3/3e_highlight_one_tcr_on_umap_cd8_tumor.svg', format='svg')
    fig.set_size_inches(old_fig_size)
else:
    print('Skipping figure 3g for faster evaluation...')

if not skip_supp_fig_4b:
    fig.set_size_inches([5, 5])
    fig.clf()
    axes = fig.subplots(2, 2)
    ind = 0
    # get the indices of the 4 desired TCRs in the figure
    search_tcrs = ['CASGRETYNEQFF', 'CASSPTGTRTEQFF', 'CASSLDSGGNTQYF', 'CATSDFGAPAEQFF']
    search_tcrs = [tcr + ' ' * (80 - len(tcr)) for tcr in search_tcrs] # null pad to match length = 80
    index_list = [df_all_tcrs.index.to_list().index(search_tcrs[0]), df_all_tcrs.index.to_list().index(search_tcrs[1]), df_all_tcrs.index.to_list().index(search_tcrs[2]), df_all_tcrs.index.to_list().index(search_tcrs[3])]
    # index_list = 9, 21, 33, 42
    target_bloodtumor = 1 # blood = 0, tumor = 1
    target_celltype = 0 # cd4 = 0, cd8 = 1
    for i, ax in enumerate(axes.flatten()):
        # Create mask for the 4 desired TCRs -------------------
        ind = index_list[i]
        ax.set_title(df_all_tcrs.index[ind].replace(' ', ''), fontsize=6)
        mask = np.logical_and(x_label[:, col_tcr] == ind, x_label[:, col_bloodtumor] == target_bloodtumor)
        mask = np.logical_and(mask, x_label[:, col_celltype] == target_celltype)
        # End of mask creation -------------------------------

        tmp = e_eval_reals[mask]
        ax.scatter(e_eval_reals[:, 0], e_eval_reals[:, 1], c='darkgray', s=1, alpha=.1)
        ax.scatter(tmp[:, 0], tmp[:, 1], s=1, c='b')
        [ax.set_xticks([]), ax.set_yticks([])]
        [ax.set_xlim(umap_lims[0]), ax.set_ylim(umap_lims[1])]
    fig.savefig(f'{figure_path}/supp_figures/4b_highlight_one_tcr_on_umap_cd4_tumor.png')
    fig.savefig(f'{figure_path}/supp_figures/4b_highlight_one_tcr_on_umap_cd4_tumor.svg', format='svg')
    fig.set_size_inches(old_fig_size)
else:
    print('Skipping supp figure 4b for faster evaluation...')

########################################
######## Testing pairwise distances for TCRs
########################################

def get_triu_pairwise_dists(data):
    return [a for a in np.triu(sklearn.metrics.pairwise_distances(data, data)).flatten().tolist() if a > 0]

mask = np.logical_and(x_label[:, col_celltype] == 1, x_label[:, col_bloodtumor] == 1)
cols_cytotoxic = [22884, 15403, 15404, 15405, 20891, 15026]
#                ['PRF1', 'GZMA', 'GZMB', 'GZMH', 'NKG7', 'GNLY']
med =  np.mean(data_rna_ambient[mask][:, cols_cytotoxic].sum(axis=-1))

# mask 1: cd8 blood, high-clonality
mask1 = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0)
mask1 = np.logical_and(mask1, x_label[:, col_bloodtumor] == 0)
mask1 = np.logical_and(mask1, x_label[:, col_celltype] == 1)

# mask 2: cd8 tumor, high-clonality, not cytotoxic
mask2 = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0)
mask2 = np.logical_and(mask2, x_label[:, col_bloodtumor] == 1)
mask2 = np.logical_and(mask2, x_label[:, col_celltype] == 1) # cd8 or cd4 indicator
mask2 = np.logical_and(mask2, data_rna_ambient[:, cols_cytotoxic].sum(axis=-1) < med)

# mask 3: cd8 tumor, high-clonality, cytotoxic
mask3 = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0)
mask3 = np.logical_and(mask3, x_label[:, col_bloodtumor] == 1)
mask3 = np.logical_and(mask3, x_label[:, col_celltype] == 1)
mask3 = np.logical_and(mask3, data_rna_ambient[:, cols_cytotoxic].sum(axis=-1) > med)

# mask 4: cd4 blood, high-clonality
mask4 = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0)
mask4 = np.logical_and(mask4, x_label[:, col_bloodtumor] == 0)
mask4 = np.logical_and(mask4, x_label[:, col_celltype] == 0)

# mask 5: cd4 tumor, high-clonality
mask5 = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0)
mask5 = np.logical_and(mask5, x_label[:, col_bloodtumor] == 1)
mask5 = np.logical_and(mask5, x_label[:, col_celltype] == 0)

#6/17/2024 - CH: get a pulled list of random and dists for both masks 2 & 3, and then plot the histogram of random vs. dists
pulled_23_list_random = []
pulled_23_list_dist = []

#1/8/2025 - Add masks for pre-defined clusters in the original paper
with open('/home/che/TRIM/data_processed/combined_data_labels_metadata.pkl', 'rb') as f:
    combined_data_labels_metadata = pickle.load(f)
cd8_tumor_mask = np.take(df_all_tcrs.fillna(0).sum(axis=1).values > 10, x_label[:, col_tcr].astype(int), axis=0)
cd8_tumor_mask = np.logical_and(cd8_tumor_mask, x_label[:, col_bloodtumor] == 1) # tumor
cd8_tumor_mask = np.logical_and(cd8_tumor_mask, x_label[:, col_celltype] == 1) # cd8
clusts = combined_data_labels_metadata[mask_preprocess]['seurat_clusters']  # data objects from another paper
assert len(clusts) == len(cd8_tumor_mask)

klf2_cd8_tumor = clusts == 1
klf2_cd8_tumor = np.logical_and(klf2_cd8_tumor, cd8_tumor_mask)
gzmk_cd8_tumor = clusts == 2
gzmk_cd8_tumor = np.logical_and(gzmk_cd8_tumor, cd8_tumor_mask)
itgae_cd8_tumor = clusts == 3
itgae_cd8_tumor = np.logical_and(itgae_cd8_tumor, cd8_tumor_mask)
cycling_cd8_tumor = clusts == 4
cycling_cd8_tumor = np.logical_and(cycling_cd8_tumor, cd8_tumor_mask)
il7r_cd8_tumor = clusts == 5
il7r_cd8_tumor = np.logical_and(il7r_cd8_tumor, cd8_tumor_mask)
isg_cd8_tumor = clusts == 6
isg_cd8_tumor = np.logical_and(isg_cd8_tumor, cd8_tumor_mask)
cd3dnk_cd8_tumor = clusts == 7
cd3dnk_cd8_tumor = np.logical_and(cd3dnk_cd8_tumor, cd8_tumor_mask)
print('KLF2+ CD8+ tumor: ', klf2_cd8_tumor.sum(), 'GZMK+ CD8+ tumor: ', gzmk_cd8_tumor.sum(), 
      'ITGAE+ CD8+ tumor: ', itgae_cd8_tumor.sum(), 'Cycling CD8+ tumor: ', cycling_cd8_tumor.sum(), 
      'IL7R+ CD8+ tumor: ', il7r_cd8_tumor.sum(), 'ISG+ CD8+ tumor: ', isg_cd8_tumor.sum(), 'CD3DNK+ CD8+ tumor: ', cd3dnk_cd8_tumor.sum())
# KLF2+ CD8+ tumor:  1739 GZMK+ CD8+ tumor:  1571 ITGAE+ CD8+ tumor:  2274 
# Cycling CD8+ tumor:  545 IL7R+ CD8+ tumor:  397 ISG+ CD8+ tumor:  147 CD3DNK+ CD8+ tumor:  52

names=['cd8_blood', 'cd8_tumor_not_cytotoxic', 'cd8_tumor_cytotoxic',
       'cd4_blood', 'cd4_tumor',
       'klf2_cd8_tumor', 'gzmk_cd8_tumor', 'itgae_cd8_tumor', 'cycling_cd8_tumor', 
       'il7r_cd8_tumor', 'isg_cd8_tumor', 'cd3dnk_cd8_tumor']

for i, mask in enumerate([mask1, mask2, mask3,
                          mask4, mask5, 
                          klf2_cd8_tumor, gzmk_cd8_tumor, itgae_cd8_tumor, cycling_cd8_tumor, 
                          il7r_cd8_tumor, isg_cd8_tumor, cd3dnk_cd8_tumor]):
    print(names[i], mask.sum())
    tmp_rna = x_rna[mask] # matrix: cells X rna_space
    tmp_tcr = pseudo_tcrs[mask] # matrix: cells (+indicator of tcr) 

    dists = []
    for tcr in np.unique(tmp_tcr):
        tmp = tmp_rna[tmp_tcr == tcr].reshape([-1, 100])
        if tmp.shape[0] < 2: continue
        tmp = get_triu_pairwise_dists(tmp) # distance in rna space
        dists.extend(tmp) # list of pairwise distances

    # np.random.seed(16)
    rng = np.random.RandomState(4)
    random = get_triu_pairwise_dists(tmp_rna[rng.choice(range(tmp_rna.shape[0]), int(1.5 * np.sqrt(len(dists))), replace=False)])

    tmpdf = pd.DataFrame(np.stack([np.concatenate([random, dists]), np.concatenate([np.zeros(len(random)), np.ones(len(dists))])], axis=-1))
    stats, pval = scipy.stats.ttest_ind(np.array(random), np.array(dists), equal_var=True, alternative='greater', axis=0)
    print(stats, pval)
    
    # Format the value into scientific notation
    coefficient, exponent = f"{pval:.3e}".split('e')
    coefficient = float(coefficient)  # Convert to float to remove trailing zeros
    exponent = int(exponent)         # Convert exponent to integer
    format_pval = f"${coefficient:.3g} \\times 10^{{{exponent}}}$"  # Use $ for LaTeX rendering

    # boxplots of random vs dists
    fig.clf()
    ax = fig.subplots(1, 1)
    # change plot size
    if i >= 5:
        fig.set_size_inches([3, 3])
    sns.boxplot(data=tmpdf, x=1, y=0, ax=ax, showfliers=False, color='w')
    ax.set_xticks([0, 1])
    # ax.set_xticklabels(['Random', 'Same clone'])
    ax.set_xticklabels(['Inter-clone', 'Intra-clone'])
    ax.set_xlim([-.5, 1.5])
    if i >= 5:
        ax.set_xlabel(f'One-sided T-test:\np-value: {format_pval}')
    else:
        # ax.set_xlabel('')
        ax.set_xlabel(f'One-sided T-test: p-value: {format_pval}')
    ax.set_ylabel('RNA pairwise distances')
    fig.tight_layout()
    fig.savefig(f'{figure_path}/supp_figures/rna_dist_sameclone_{names[i]}.png', dpi=300)
    fig.savefig(f'{figure_path}/supp_figures/rna_dist_sameclone_{names[i]}.svg', format='svg')

    # histograms of random vs dists
    plt.figure(figsize=(10, 6))
    sns.kdeplot(random, label='Inter-clone', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(dists, label='Intra-clone', color='red', fill=True, alpha=0.5)
    plt.xlabel('RNA pairwise distances')
    plt.ylabel('Density')
    # plt.title('Intra-clone vs inter-clone distances amongst high clonality TCRs in CD8 tumor')
    plt.legend(loc='upper right')
    # plt.savefig(f'tcr/git/tcr/figures/rna_dist_sameclone_dist_{names[i]}.png')

    # append mask 2 and 3 results
    if i == 1 or i == 2:
        pulled_23_list_random.extend(random)
        pulled_23_list_dist.extend(dists)

# plot Histograms of pulled_23_list_random and pulled_23_list_distplt.figure(figsize=(10, 6))
sns.kdeplot(pulled_23_list_random, label='Inter-clone', color='blue', fill=True, alpha=0.5)
sns.kdeplot(pulled_23_list_dist, label='Intra-clone', color='red', fill=True, alpha=0.5)
plt.xlabel('RNA pairwise distances')
plt.ylabel('Density')
# plt.title('Intra-clone vs inter-clone distances amongst high clonality TCRs in CD8 tumor')
plt.legend(loc='upper right')
# plt.savefig('tcr/figures/rna_dist_sameclone_dist_dist_pulled_cd8_tumor.png')

# plot boxplot of pulled_23_list_random and pulled_23_list_dist
tmpdf = pd.DataFrame(np.stack([np.concatenate([pulled_23_list_random, pulled_23_list_dist]), np.concatenate([np.zeros(len(pulled_23_list_random)), np.ones(len(pulled_23_list_dist))])], axis=-1))
print(scipy.stats.ttest_ind(np.array(random), np.array(dists), equal_var=True, alternative='greater', axis=0))
fig.clf()
ax = fig.subplots(1, 1)
sns.boxplot(data=tmpdf, x=1, y=0, ax=ax, showfliers=False, color='w')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Random', 'Same clone'])
ax.set_xlim([-.5, 1.5])
ax.set_xlabel('')
ax.set_ylabel('RNA pairwise dist')
# fig.savefig('tcr/figures/rna_dist_sameclone_pulled_cd8_tumor.png'.format(i))


#########################
# figure 3f

# read in combined_data_labels_metadata from pickle file (generated in train_contrastive_embedding.py)
with open('/home/che/TRIM/data_processed/combined_data_labels_metadata.pkl', 'rb') as f:
    combined_data_labels_metadata = pickle.load(f)

# CH: clusters from paper cd4blood old ----------------
mask = x_label[:, col_celltype] == 0 #get cd4
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 0) #subset to blood
r = np.random.choice(range(mask.sum()), mask.sum(), replace=False)
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters'] #data objects from another paper
fig.clf()
ax = fig.subplots(1, 1)
scat = ax.scatter(e_eval_reals[mask][r, 0], e_eval_reals[mask][r, 1], s=1, cmap=mpl.cm.tab10, c=clusts[r])
ax.legend(handles=scat.legend_elements()[0], labels=['CCR7+ CD4', 'II7R+ CD4', 'GZMK+ CD4', 'HLA-DR+ CD4',
                                                     'FOXP3+ CD4', 'RPS CD4', 'ISG CD4', 'GZM+ CD4', 'Cycling CD4'], loc='lower right')
[ax.set_xticks([]), ax.set_yticks([])]
# fig.savefig('tcr/figures/clusters_from_paper_cd4blood_old.png')

# CH: clusters from paper cd4blood new (include all dots) ----------------
mask_all = np.isin(x_label[:, col_celltype], [0, 1])  # get cd4 and cd8
mask_all = np.logical_and(mask_all, np.isin(x_label[:, col_bloodtumor], [0, 1]))  # get tumor and blood
r_all = np.random.choice(range(mask_all.sum()), mask_all.sum(), replace=False)
mask = x_label[:, col_celltype] == 0  # get cd4
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 0)  # subset to blood
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters']  # data objects from another paper
clusts_all = combined_data_labels_metadata[mask_preprocess][mask_all]['seurat_clusters']  # data objects from another paper
clusts_all.loc[~clusts_all.index.isin(clusts.index)] = np.nan
# Create a new figure and axis
fig, ax = plt.subplots(1, 1)
# Plotting NaN values as light grey
nan_mask = np.isnan(clusts_all[r_all])
ax.scatter(e_eval_reals[mask_all][r_all, 0][nan_mask], e_eval_reals[mask_all][r_all, 1][nan_mask], s=1, c='lightgrey')
# Plotting non-NaN values
non_nan_mask = ~np.isnan(clusts_all[r_all])
scat = ax.scatter(e_eval_reals[mask_all][r_all, 0][non_nan_mask], e_eval_reals[mask_all][r_all, 1][non_nan_mask],
                   s=1, cmap=mpl.cm.tab10, c=clusts_all[r_all][non_nan_mask])
# Adding legend without NaN label
legend_labels = ['CCR7+ CD4', 'II7R+ CD4', 'GZMK+ CD4', 'HLA-DR+ CD4', 'FOXP3+ CD4', 'RPS CD4', 'ISG CD4', 'GZM+ CD4', 'Cycling CD4']
handles, _ = scat.legend_elements()
ax.legend(handles=handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
# Removing ticks
ax.set_xticks([])
ax.set_yticks([])
# Adjust layout to make room for the legend
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
# fig.savefig('tcr/figures/clusters_from_paper_cd4blood.png')  # PNG for high resolution
# fig.savefig('tcr/figures/clusters_from_paper_cd4blood.pdf', format='pdf')  # EPS for vector format

# CH: clusters from paper cd8blood old ----------------
mask = x_label[:, col_celltype] == 1 #get cd8
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 0) #subset to blood
r = np.random.choice(range(mask.sum()), mask.sum(), replace=False)
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters'] #data objects from another paper
fig.clf()
ax = fig.subplots(1, 1)
scat = ax.scatter(e_eval_reals[mask][r, 0], e_eval_reals[mask][r, 1], s=1, cmap=mpl.cm.tab10, c=clusts[r])
ax.legend(handles=scat.legend_elements()[0], labels=['GZMK+ CD8', 'GZMB CD8', 'CCR7+ CD8', 'FGFBP2 CD8',
                                                     'IL7R+ CD8', 'CD38+ CD8', 'DN T cells', 'LTB+ CD8', 'KLRB1+ CD8'], 
                                                     loc='lower right')
[ax.set_xticks([]), ax.set_yticks([])]
# fig.savefig('tcr/figures/clusters_from_paper_cd8blood_old.png')

# CH: clusters from paper cd8blood new ----------------
mask_all = np.isin(x_label[:, col_celltype], [0, 1]) #get cd4 and cd8
mask_all = np.logical_and(mask_all, np.isin(x_label[:, col_bloodtumor], [0,1])) #get tumor and blood
r_all = np.random.choice(range(mask_all.sum()), mask_all.sum(), replace=False)
mask = x_label[:, col_celltype] == 1  # get cd8
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 0)  # subset to blood
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters']  # data objects from another paper
clusts_all = combined_data_labels_metadata[mask_preprocess][mask_all]['seurat_clusters']  # data objects from another paper
clusts_all.loc[~clusts_all.index.isin(clusts.index)] = np.nan
# Create a new figure and axis
fig, ax = plt.subplots(1, 1)
# Plotting NaN values as light grey
nan_mask = np.isnan(clusts_all[r_all])
ax.scatter(e_eval_reals[mask_all][r_all, 0][nan_mask], e_eval_reals[mask_all][r_all, 1][nan_mask], s=1, c='lightgrey')
# Plotting non-NaN values
non_nan_mask = ~np.isnan(clusts_all[r_all])
scat = ax.scatter(e_eval_reals[mask_all][r_all, 0][non_nan_mask], e_eval_reals[mask_all][r_all, 1][non_nan_mask],
                   s=1, cmap=mpl.cm.tab10, c=clusts_all[r_all][non_nan_mask])
# Adding legend without NaN label
legend_labels = ['GZMK+ CD8', 'GZMB CD8', 'CCR7+ CD8', 'FGFBP2 CD8', 'IL7R+ CD8', 'CD38+ CD8', 'DN T cells', 'LTB+ CD8', 'KLRB1+ CD8']
handles, _ = scat.legend_elements()
ax.legend(handles=handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
# Removing ticks
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
# Saving the figure
# fig.savefig('tcr/figures/clusters_from_paper_cd8blood.png')
# fig.savefig('tcr/figures/clusters_from_paper_cd8blood.pdf', format='pdf')


# CH: clusters from paper cd4tumor (old) ----------------
mask = x_label[:, col_celltype] == 0 #get cd4
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 1) #subset to tumor
r = np.random.choice(range(mask.sum()), mask.sum(), replace=False)
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters'] #data objects from another paper
fig.clf()
ax = fig.subplots(1, 1)
scat = ax.scatter(e_eval_reals[mask][r, 0], e_eval_reals[mask][r, 1], s=1, cmap=mpl.cm.tab10, c=clusts[r])
ax.legend(handles=scat.legend_elements()[0], labels=['IL7R+ CD4', 'OX40+ Treg', 'HLA-DR+ Treg', 'CXCL13+ CD4', 
                                                     'Cycling CD4', 'GZMA+ CD4', 'ISG CD4'], loc='lower left')
[ax.set_xticks([]), ax.set_yticks([])]
# fig.savefig('tcr/figures/clusters_from_paper_cd4tumor_old.png')

# CH: clusters from paper cd4tumor (new) ----------------
mask_all = np.isin(x_label[:, col_celltype], [0, 1]) #get cd4 and cd8
mask_all = np.logical_and(mask_all, np.isin(x_label[:, col_bloodtumor], [0,1])) #get tumor and blood
r_all = np.random.choice(range(mask_all.sum()), mask_all.sum(), replace=False)
mask = x_label[:, col_celltype] == 0  # get cd4
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 1)  # subset to tumor
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters']  # data objects from another paper
clusts_all = combined_data_labels_metadata[mask_preprocess][mask_all]['seurat_clusters']  # data objects from another paper
clusts_all.loc[~clusts_all.index.isin(clusts.index)] = np.nan
# Create a new figure and axis
fig, ax = plt.subplots(1, 1)
# Plotting NaN values as light grey
nan_mask = np.isnan(clusts_all[r_all])
ax.scatter(e_eval_reals[mask_all][r_all, 0][nan_mask], e_eval_reals[mask_all][r_all, 1][nan_mask], s=1, c='lightgrey')
# Plotting non-NaN values
non_nan_mask = ~np.isnan(clusts_all[r_all])
scat = ax.scatter(e_eval_reals[mask_all][r_all, 0][non_nan_mask], e_eval_reals[mask_all][r_all, 1][non_nan_mask],
                   s=1, cmap=mpl.cm.tab10, c=clusts_all[r_all][non_nan_mask])
# Adding legend without NaN label
legend_labels = ['IL7R+ CD4', 'OX40+ Treg', 'HLA-DR+ Treg', 'CXCL13+ CD4', 'Cycling CD4', 'GZMA+ CD4', 'ISG CD4']
handles, _ = scat.legend_elements()
ax.legend(handles=handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
# Removing ticks
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
# Saving the figure
# fig.savefig('tcr/figures/clusters_from_paper_cd4tumor.png')
# fig.savefig('tcr/figures/clusters_from_paper_cd4tumor.pdf', format='pdf')


# clusters from paper cd8tumor (old) ----------------
mask = x_label[:, col_celltype] == 1 #get cd8
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 1) #subset to tumor
r = np.random.choice(range(mask.sum()), mask.sum(), replace=False)
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters'] #data objects from another paper
fig.clf()
ax = fig.subplots(1, 1)
scat = ax.scatter(e_eval_reals[mask][r, 0], e_eval_reals[mask][r, 1], s=1, cmap=mpl.cm.tab10, c=clusts[r])
ax.legend(handles=scat.legend_elements()[0], labels=['KLF2+ CD8', 'GZMK+ CD8', 'ITGAE+ CD8', 'Cycling CD8', 'IL7R+ CD8', 'ISG CD8', 'CD3D- NK'], loc='upper left')
[ax.set_xticks([]), ax.set_yticks([])]
fig.savefig(f'{figure_path}/figure3/3f_clusters_from_paper_cd8tumor.png')
fig.savefig(f'{figure_path}/figure3/3f_clusters_from_paper_cd8tumor.svg', format='svg')

# clusters from paper cd8tumor (new) ----------------
mask_all = np.isin(x_label[:, col_celltype], [0, 1]) #get cd4 and cd8
mask_all = np.logical_and(mask_all, np.isin(x_label[:, col_bloodtumor], [0,1])) #get tumor and blood
r_all = np.random.choice(range(mask_all.sum()), mask_all.sum(), replace=False)
mask = x_label[:, col_celltype] == 1  # get cd8
mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 1)  # subset to tumor
clusts = combined_data_labels_metadata[mask_preprocess][mask]['seurat_clusters']  # data objects from another paper
clusts_all = combined_data_labels_metadata[mask_preprocess][mask_all]['seurat_clusters']  # data objects from another paper
clusts_all.loc[~clusts_all.index.isin(clusts.index)] = np.nan
# Create a new figure and axis
fig, ax = plt.subplots(1, 1)
# Plotting NaN values as light grey
nan_mask = np.isnan(clusts_all[r_all])
ax.scatter(e_eval_reals[mask_all][r_all, 0][nan_mask], e_eval_reals[mask_all][r_all, 1][nan_mask], s=1, c='lightgrey')
# Plotting non-NaN values
non_nan_mask = ~np.isnan(clusts_all[r_all])
scat = ax.scatter(e_eval_reals[mask_all][r_all, 0][non_nan_mask], e_eval_reals[mask_all][r_all, 1][non_nan_mask],
                   s=1, cmap=mpl.cm.tab10, c=clusts_all[r_all][non_nan_mask])
# Adding legend without NaN label
legend_labels = ['KLF2+ CD8', 'GZMK+ CD8', 'ITGAE+ CD8', 'Cycling CD8', 'IL7R+ CD8', 'ISG CD8', 'CD3D- NK']
handles, _ = scat.legend_elements()
ax.legend(handles=handles, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
# Removing ticks
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
# Saving the figure
fig.savefig(f'{figure_path}/figure3/clusters_from_paper_cd8tumor_v1.png')
fig.savefig(f'{figure_path}/figure3/clusters_from_paper_cd8tumor_v1.svg', format='svg')

cols_cytotoxic = [22884, 15403, 15404, 15405, 20891, 15026]
#                ['PRF1', 'GZMA', 'GZMB', 'GZMH', 'NKG7', 'GNLY']
c = data_rna_ambient[mask][r][:, cols_cytotoxic].sum(axis=-1)
med =  np.mean(data_rna_ambient[mask][:, cols_cytotoxic].sum(axis=-1))
fig.clf()
ax = fig.subplots(1, 1)
scat = ax.scatter(e_eval_reals[mask][r, 0], e_eval_reals[mask][r, 1], s=1, cmap=mpl.cm.bwr, c=c, vmin=np.percentile(c, 5), vmax=np.percentile(c, 95))
[ax.set_xticks([]), ax.set_yticks([])]
fig.savefig(f'{figure_path}/figure3/3f_cytotoxic_from_paper_cd8tumor.png')
fig.savefig(f'{figure_path}/figure3/3f_cytotoxic_from_paper_cd8tumor.svg', format='svg')


