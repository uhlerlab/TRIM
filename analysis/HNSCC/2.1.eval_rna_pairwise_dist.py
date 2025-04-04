import os
import numpy as np
import random
random.seed(0)
np.random.seed(0)

# os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=32
# os.environ["PYTHONHASHSEED"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

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
# args.e_eval_reals_path = '/home/che/TRIM/data_figures/HNSCC/e_eval_reals.pickle'
# if os.path.exists(args.e_eval_reals_path):
#     if load_precalculated_umap_real:
#         with open(args.e_eval_reals_path, 'rb') as f:
#             e_eval_reals = pickle.load(f)
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         print('!!!!!!! Loaded e_eval_reals.pickle from', args.e_eval_reals_path)
#         print('!!!!!!! if the processed data is updated, please delete the file and re-run to generate a new e_eval_reals.pickle')
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     else:
#         raise ValueError('Found e_eval_reals.pickle at {} but load_precalculated_umap_real is set to False. Please delete the file and re-run to generate a new e_eval_reals.pickle'.format(args.e_eval_reals_path))
# else:
#     e_eval_reals = viz_reducer.transform(data_rna)
#     with open(args.e_eval_reals_path, 'wb') as f:
#         pickle.dump(e_eval_reals, f)
#     print('Saved e_eval_reals.pickle to', args.e_eval_reals_path)
# print('Finished pca/viz_umap data in {:.1f} s'.format(time.time() - t))

#############################################
#############################################
# load model
print('Starting to load model...')
t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_rna = data_rna
x_tcr = data_tcr
x_label = data_labels

mask_preprocess = x_label[:, col_tcr] != ID_NULL_TCR
x_rna = x_rna[mask_preprocess]
x_tcr = x_tcr[mask_preprocess]
x_label = x_label[mask_preprocess]
# e_eval_reals = e_eval_reals[mask_preprocess]

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

#########################
# figure 3b
# rna pairwise dist vs hamming pairwise dist

# mask = np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_celltype] == 1)
rna_dists_by_hamming = {}
tmp_data = x_rna
tmp_labels = x_label

base_sample_rate = 1

# rna_dists = []
# for i in range(df_all_tcrs.shape[0]):
#     if i == 0: continue
#     if i % 5000 == 0: print(i)

#     if i % base_sample_rate != 0: continue
#     tmp = tmp_data[tmp_labels[:, col_tcr] == i]

#     if tmp.shape[0] < 2: continue
#     rna_dists.append(sklearn.metrics.pairwise_distances(tmp, tmp).mean())
# rna_dists_by_hamming[0] = rna_dists

vocab = set()
[[vocab.add(c) for c in l] for l in df_all_tcrs.index] 
vocab_char2num = {v: i for i, v in enumerate(sorted(vocab))}
vocab_num2char = {i: v for i, v in enumerate(sorted(vocab))}
df_all_tcrs_array = np.array([[vocab_char2num[char] for char in i] for i in df_all_tcrs.index])

check_type = 'same_pt_source_type_prepost' # 'all', 'same_pt', 'same_pt_source', 'same_pt_source_type', 'same_pt_source_type_prepost'
test_expanded = True # set this to True unless check_type == 'all
print('Check type:', check_type)
print('Test expanded:', test_expanded)
for hamming_distance in range(0, 11):
    sample_rate = base_sample_rate * (1 + hamming_distance // 2)
    rna_dists = []
    print("Hamming distance {}".format(hamming_distance))
    for i in range(df_all_tcrs.shape[0]): # loop over all unique TCRs
        if i == 0: continue
        if i % 5000 == 0: print(i)

        if i % sample_rate != 0: continue

        if test_expanded is True:
            if df_all_tcrs.iloc[i].sum() <= 1: continue
        elif test_expanded is False:
            if df_all_tcrs.iloc[i].sum() > 1: continue

        dists = (df_all_tcrs_array[i][np.newaxis, :] != df_all_tcrs_array).sum(axis=-1)
        rows = np.argwhere(dists == hamming_distance)
        if rows.shape[0] > 0:
            mask_irna = tmp_labels[:, col_tcr] == i
            i_rna = tmp_data[mask_irna, :]
            if i_rna.shape[0] == 0: continue
            if i_rna.shape[0] == 1 and (hamming_distance == 0): continue

            for iter_j, j in enumerate(rows):
                if j < i: continue
                if iter_j % sample_rate != 0: continue

                mask_jrna = tmp_labels[:, col_tcr] == j
                j_rna = tmp_data[mask_jrna, :]

                mask_ij_match_labels = np.ones(j_rna.shape[0]).astype(bool)

                if check_type == 'same_pt':
                    # select one patient if tcr i is in multiple patients and subset irna and jrna to that patient
                    current_pt = tmp_labels[mask_irna, col_patient][0]
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_patient] == current_pt)
                    i_rna = tmp_data[mask_irna, :]
                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_patient] == current_pt)
                elif check_type == 'same_pt_source':
                    # select one patient and one source if tcr i is in multiple patients and sources and subset irna and jrna to that patient and source
                    current_pt = tmp_labels[mask_irna, col_patient][0]
                    current_source = tmp_labels[mask_irna, col_bloodtumor][0]
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_patient] == current_pt)
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_bloodtumor] == current_source)
                    i_rna = tmp_data[mask_irna, :]

                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_patient] == current_pt)
                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_bloodtumor] == current_source)
                elif check_type == 'same_pt_source_type':
                    # select one patient and one source and one type if tcr i is in multiple patients and sources and types and subset irna and jrna to that patient and source and type
                    current_pt = tmp_labels[mask_irna, col_patient][0]
                    current_source = tmp_labels[mask_irna, col_bloodtumor][0]
                    current_type = tmp_labels[mask_irna, col_celltype][0]
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_patient] == current_pt)
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_bloodtumor] == current_source)
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_celltype] == current_type)
                    i_rna = tmp_data[mask_irna, :]

                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_patient] == current_pt)
                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_bloodtumor] == current_source)
                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_celltype] == current_type)
                elif check_type == 'same_pt_source_type_prepost':
                    # select one patient and one source and one type and one prepost if tcr i is in multiple patients and sources and types and prepost and subset irna and jrna to that patient and source and type and prepost
                    current_pt = tmp_labels[mask_irna, col_patient][0]
                    current_source = tmp_labels[mask_irna, col_bloodtumor][0]
                    current_type = tmp_labels[mask_irna, col_celltype][0]
                    current_prepost = tmp_labels[mask_irna, col_prepost][0]
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_patient] == current_pt)
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_bloodtumor] == current_source)
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_celltype] == current_type)
                    mask_irna = np.logical_and(mask_irna, tmp_labels[:, col_prepost] == current_prepost)
                    i_rna = tmp_data[mask_irna, :]

                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_patient] == current_pt)
                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_bloodtumor] == current_source)
                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_celltype] == current_type)
                    mask_ij_match_labels = np.logical_and(mask_ij_match_labels, tmp_labels[mask_jrna, col_prepost] == current_prepost)

                j_rna = j_rna[mask_ij_match_labels]

                if j_rna.shape[0] == 0: continue
                # if j_rna.shape[0] == 1: continue

                if test_expanded is True:
                    if df_all_tcrs.iloc[j[0]].sum() <= 1: continue
                elif test_expanded is False:
                    if df_all_tcrs.iloc[j[0]].sum() > 1: continue

                # add a double-check here
                cdr3_i = df_all_tcrs.iloc[i].name.strip()
                cdr3_j = df_all_tcrs.iloc[j[0]].name.strip()
                a, b = cdr3_i, cdr3_j
                a, b = a.ljust(max(len(a), len(b))), b.ljust(max(len(a), len(b)))
                hamming_dist = sum(x != y for x, y in zip(a, b))
                assert hamming_dist == hamming_distance, f'Hamming distance mismatch: {hamming_dist} != {hamming_distance} for TCRs {a} and {b}. This should not happen.'

                dists = sklearn.metrics.pairwise_distances(i_rna, j_rna)
                rna_dists.append(np.mean(dists))
    rna_dists_by_hamming[hamming_distance] = rna_dists
    print("Mean: {:.3f}".format(np.mean(rna_dists)))

fig.set_size_inches([9, 4])
fig.clf()
ax = fig.subplots(1, 1)
min_points = min([len(rna_dists_by_hamming[key]) for key in rna_dists_by_hamming])
for hamming_distance in sorted(rna_dists_by_hamming.keys()):
    pts = rna_dists_by_hamming[hamming_distance]
    if len(pts) == 0: continue
    mean_ = np.mean(pts)

    r = np.random.choice(range(len(pts)), min(500, len(pts)), replace=False)
    ax.scatter(len(r) * [hamming_distance], np.array(pts)[r], c='b', s=1, alpha=.5)
    ax.plot([hamming_distance - .25, hamming_distance + .25], 2 * [mean_], c='k', linestyle='--')
ax.set_xlabel('Hamming pairwise distance')
ax.set_ylabel('RNA pairwise distance')
ax.set_xticks(range(0, max(rna_dists_by_hamming.keys()) + 1))
ax.set_xlim([-1, max(rna_dists_by_hamming.keys()) + 1])
ax.set_ylim([0, ax.get_ylim()[1]])
# fig.savefig('figures/hamming_pairwise_by_rna_pairwise_{}exp.png'.format('' if test_expanded else 'non'))
# fig.savefig('figures/tmp.png')
fig.savefig(f"{figure_path}/figure3/3b_hamming_pairwise_by_rna_pairwise_{'' if test_expanded else 'non'}exp{'' if check_type == 'all' else check_type}.png")
fig.savefig(f"{figure_path}/figure3/3b_hamming_pairwise_by_rna_pairwise_{'' if test_expanded else 'non'}exp{'' if check_type == 'all' else check_type}.svg", format='svg')
fig.set_size_inches(old_fig_size)


fig.clf()
ax = fig.subplots(1, 1)
keys = sorted([k for k in rna_dists_by_hamming.keys() if k != 0])
vals = [np.mean(rna_dists_by_hamming[key]) for key in keys]
ax.scatter(keys, vals, c='b')
ax.plot(keys, vals, c='b')
ax.set_title('Mean', fontsize=8)
[ax.set_ylabel('Distance in RNA', fontsize=6) for ax in [ax]]
[ax.set_xticks(keys) for ax in [ax]]
ax.set_xlabel('Hamming Distance', fontsize=6)
ax.set_xticks(range(0, max(rna_dists_by_hamming.keys()) + 1))
# fig.savefig('figures/hamming_pairwise_by_rna_pairwise_means_{}exp.png'.format('' if test_expanded else 'non'))
# fig.savefig('figures/tmp.png')
fig.savefig(f"{figure_path}/figure3/3b_hamming_pairwise_by_rna_pairwise_means_{'' if test_expanded else 'non'}exp{'' if check_type == 'all' else check_type}.png")
fig.savefig(f"{figure_path}/figure3/3b_hamming_pairwise_by_rna_pairwise_means_{'' if test_expanded else 'non'}exp{'' if check_type == 'all' else check_type}.svg", format='svg')
