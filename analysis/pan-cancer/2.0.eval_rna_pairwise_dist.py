import os
import numpy as np
import random
random.seed(0)
np.random.seed(0)

os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=32
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import glob
import sklearn
import sklearn.decomposition
import sklearn.cluster
from sklearn.svm import SVC
import time
import umap
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import torch
import time
import sys
import json
import statsmodels.api as sm
import argparse
from torch import nn
import torch.nn.functional as F
fig = plt.figure()
old_fig_size = fig.get_size_inches()

######################################################################
######################################################################
######################################################################

data_path = '/data/che/TRIM/panc' #/data/che/panc
fig_path = '/home/che/TRIM/git/tcr/figures_pancancer'

######################################################################
######################################################################
######################################################################
# analysis figures

def make_legend(ax, labels, s=20, cmap=mpl.cm.jet, **kwargs):
    numlabs = len(labels)
    for i, label in enumerate(labels):
        if numlabs > 1:
            ax.scatter(0, 0, s=s, c=[cmap(1 * i / (numlabs - 1))], label=label)
        else:
            ax.scatter(0, 0, s=s, c=[cmap(1.)], label=label)
    ax.scatter(0, 0, s=2 * s, c='w')
    ax.legend(**kwargs)

args_folder = f'{data_path}/model/args.txt'
args = argparse.ArgumentParser().parse_args()
with open(args_folder, 'r') as f:
    args.__dict__ = json.load(f)

npca = 100
with open(f'{data_path}/data_pca.npz', 'rb') as f:
    npzfile = np.load(f, allow_pickle=True)
    df = pd.DataFrame(npzfile['df'], index=npzfile['df_index'], columns=npzfile['df_columns'])

mask_null = df['cdr3'] != 'None'
mask_patient = df['patient'].apply(lambda tmp: tmp in ['ESCA.P20181123', 'ESCA.P20190404', 'ESCA.P20190410', 'ESCA.P20190411', 'ESCA.P20190613',
                                                       'THCA.P20181226', 'THCA.P20190108', 'THCA.P20190118', 'THCA.P20190122', 'THCA.P20190125', 'THCA.P20190612', 'THCA.P20190730', 'THCA.P20190816',
                                                       'UCEC.P20181122', 'UCEC.P20190213', 'UCEC.P20190305', 'UCEC.P20190312', 'UCEC.P20190717', 'UCEC.P20190911']).values
mask = np.logical_and(mask_null, mask_patient)
df = df[mask]

df_tcr_counts = df['cdr3'].value_counts()
df['cdr3_count'] = np.array([df_tcr_counts.loc[tcr] for tcr in df['cdr3'].tolist()])

dict_loc2num = {'P': 0, 'N': 0, 'T': 1,}
dict_patient2num = {p: i for i, p in enumerate(sorted(df['patient'].unique()))}
dict_cancertype2num = {c: i for i, c in enumerate(sorted(df['cancerType'].unique()))}
df['loc'] = df['loc'].apply(lambda tmp: dict_loc2num[tmp])
df['patient'] = df['patient'].apply(lambda tmp: dict_patient2num[tmp])
df['cancerType'] = df['cancerType'].apply(lambda tmp: dict_cancertype2num[tmp])
col_loc = df.columns[npca:].tolist().index('loc')
col_patient = df.columns[npca:].tolist().index('patient')
col_cancertype = df.columns[npca:].tolist().index('cancerType')

df_all_tcrs = pd.DataFrame(df['cdr3'].unique())
tcr_max_len = max(df_all_tcrs.iloc[:, 0].apply(lambda tmp: len(tmp)))
df_all_tcrs.iloc[:, 0] = df_all_tcrs.iloc[:, 0].apply(lambda tmp: tmp.ljust(tcr_max_len))

def one_hot_along_3rd_axis(x):
    out = np.zeros([x.shape[0], x.shape[1], len(vocab)])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j, x[i, j]] = 1
    return out

vocab = set()
[[vocab.add(c) for c in l] for l in df_all_tcrs.iloc[:, 0]]
vocab_char2num = {v: i for i, v in enumerate(sorted(vocab))}
vocab_num2char = {i: v for i, v in enumerate(sorted(vocab))}
df_all_tcrs_array = np.array([[vocab_char2num[char] for char in i] for i in df_all_tcrs.iloc[:, 0]])
tcr_max_len = df_all_tcrs_array.shape[1]
print(sorted(vocab))
df_all_tcrs_array = one_hot_along_3rd_axis(df_all_tcrs_array) #(num_unique_tcrs, max_seq_length, vocab_size)

df_preds = df.copy()
df_preds['pseudoclone'] = df_preds['cdr3']
for patient in range(len(dict_patient2num.keys())):
    with open(f"{data_path}/model/preds_patient{patient}.npz", 'rb') as f:
        npzfile = np.load(f, allow_pickle=True)
        preds_rna = npzfile['preds_rna']
        pseudo_tcrs = npzfile['pseudo_tcrs']
    df_preds.iloc[df_preds['patient'] == patient, :npca] = preds_rna
    df_preds.iloc[df_preds['patient'] == patient, -1] = pseudo_tcrs + (df_preds.shape[0] * patient)

# df_pseudoclone_counts = df_preds['pseudoclone'].value_counts()
# df_preds['pseudoclone_count'] = np.array([df_pseudoclone_counts.loc[tcr] for tcr in df_preds['pseudoclone'].tolist()])

preds_rna = df_preds.iloc[:, :npca].values
pseudo_tcrs = df_preds['pseudoclone'].values

# with open(os.path.join(args.output_folder, 'preds.npz'), 'rb') as f:
#     npzfile = np.load(f, allow_pickle=True)
#     preds_rna = npzfile['preds_rna']
#     preds_tcr = npzfile['preds_tcr']
#     pseudo_tcrs = npzfile['pseudo_tcrs']
#     mask_train = npzfile['mask_train']

pseudo_tcrs_unique = np.unique(pseudo_tcrs, return_counts=True)
pseudo_tcrs_unique = pd.DataFrame(pseudo_tcrs_unique[1], index=pseudo_tcrs_unique[0])
pseudo_tcr_counts = np.array([pseudo_tcrs_unique.loc[tcr] for tcr in pseudo_tcrs.tolist()])

# add in cellType labels for plotting
df_cellType = pd.read_csv(f'{data_path}/data_cellType.csv', index_col=0)
df = df.merge(df_cellType, left_index=True, right_index=True, how='left')

######################################################################
# to make hamming vs rna charts
unique_tcrs_list = df_all_tcrs.iloc[:, 0].str.strip().tolist()
tcrs_id = df['cdr3'].apply(lambda tmp: unique_tcrs_list.index(tmp))
df_all_tcrs_array_2d = df_all_tcrs_array.argmax(axis=-1) # (num_tcrs, max_seq_len)

# Note: to map indices to cancerType: dict_cancertype2num
# cancer_index = [0, 1, 2]
for cancer_index in [[0], [1], [2], [0, 1, 2]]:
    if len(cancer_index) == 1:
        cancer_type = [k for k, v in dict_cancertype2num.items() if v == cancer_index[0]][0]
    elif len(cancer_index) == 3:
        cancer_type = 'all'
    mask_cancer = df['cancerType'].isin(cancer_index)
    # mask_cancer = np.ones(df.shape[0]).astype(bool)
    rna_dists_by_hamming = {}
    tmp_data = df.iloc[:, :npca].values[mask_cancer]
    tmp_labels = tcrs_id[mask_cancer]
    tmp_counts = df['cdr3_count'][mask_cancer]

    base_sample_rate = 5

    rna_dists = []
    for i in range(df_all_tcrs.shape[0]):
        if i == 0: continue
        if i % 5000 == 0: print(i)

        if i % base_sample_rate != 0: continue
        tmp = tmp_data[tmp_labels == i]
        # tmp = tmp.iloc[:, mask_hvg].values
        if tmp.shape[0] < 2: continue
        rna_dists.append(sklearn.metrics.pairwise_distances(tmp, tmp).mean())
    rna_dists_by_hamming[0] = rna_dists

    # test_expanded = True
    for hamming_distance in range(1, 11):
        sample_rate = base_sample_rate * (1 + hamming_distance // 2)
        rna_dists = []
        print("Hamming distance {}".format(hamming_distance))
        for i in range(df_all_tcrs.shape[0]):
            if i == 0: continue
            if i % 5000 == 0: print(i)

            if i % sample_rate != 0: continue

            # if test_expanded:
            #     if df_tcr_counts.iloc[i] <= 1: continue
            # else:
            #     if df_tcr_counts.iloc[i] > 1: continue

            dists = (df_all_tcrs_array_2d[i][np.newaxis, :] != df_all_tcrs_array_2d).sum(axis=-1)
            rows = np.argwhere(dists == hamming_distance).reshape([-1])
            if rows.shape[0] > 0:
                i_rna = tmp_data[tmp_labels == i, :]
                if i_rna.shape[0] == 0: continue
                # if i_rna.shape[0] == 1: continue

                for iter_j, j in enumerate(rows):
                    if j < i: continue
                    if iter_j % sample_rate != 0: continue

                    j_rna = tmp_data[tmp_labels == j, :]
                    if j_rna.shape[0] == 0: continue
                    # if j_rna.shape[0] == 1: continue

                    # if test_expanded:
                    #     if df_tcr_counts.iloc[j] <= 1: continue
                    # else:
                    #     if df_tcr_counts.iloc[j] > 1: continue

                    # add a check here
                    cell_i = tmp_labels[tmp_labels == i].index.tolist()
                    cell_j = tmp_labels[tmp_labels == j].index.tolist()
                    cdr3_i = df[df.index.isin(cell_i)]['cdr3'].unique()
                    cdr3_j = df[df.index.isin(cell_j)]['cdr3'].unique()
                    if len(cdr3_i) == 1 and len(cdr3_j) == 1:
                        a, b = cdr3_i[0], cdr3_j[0]
                        a, b = a.ljust(max(len(a), len(b))), b.ljust(max(len(a), len(b)))
                        hamming_dist = sum(x != y for x, y in zip(a, b))
                        assert hamming_dist == hamming_distance, f'Hamming distance mismatch: {hamming_dist} != {hamming_distance} for TCRs {a} and {b}. This should not happen.'
                    else:
                        raise ValueError(f'Unexpected case: {len(cdr3_i)} != 1 or {len(cdr3_j)} != 1 for TCRs {cdr3_i} and {cdr3_j}. This should not happen.')
                    dists = sklearn.metrics.pairwise_distances(i_rna, j_rna)
                    rna_dists.append(np.mean(dists))
        rna_dists_by_hamming[hamming_distance] = rna_dists
        print("Mean: {:.3f}".format(np.mean(rna_dists)))

    fig.set_size_inches([9, 4])
    fig.clf()
    ax = fig.subplots(1, 1)
    np.random.seed(0)
    min_points = min([len(rna_dists_by_hamming[key]) for key in rna_dists_by_hamming])
    for hamming_distance in sorted(rna_dists_by_hamming.keys()):
        pts = rna_dists_by_hamming[hamming_distance]
        if len(pts) == 0: continue
        mean_ = np.mean(pts)

        r = np.random.choice(range(len(pts)), min(100, len(pts)), replace=False)
        ax.scatter(len(r) * [hamming_distance], np.array(pts)[r], c='b', s=1, alpha=.5)
        ax.plot([hamming_distance - .25, hamming_distance + .25], 2 * [mean_], c='k', linestyle='--')
    ax.set_xlabel('Hamming pairwise distance')
    ax.set_ylabel('RNA pairwise distance')
    ax.set_xticks(range(0, max(rna_dists_by_hamming.keys()) + 1))
    ax.set_xlim([-1, max(rna_dists_by_hamming.keys()) + 1])
    ax.set_ylim([0, ax.get_ylim()[1]])
    fig.savefig(f'{fig_path}/rna_distance_vs_tcr_hamming_distance_{cancer_type}.png', dpi=300)
    fig.savefig(f'{fig_path}/rna_distance_vs_tcr_hamming_distance_{cancer_type}.svg', format='svg')
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
    fig.savefig(f'{fig_path}/mean_rna_distance_vs_tcr_hamming_distance_{cancer_type}.png', dpi=300)
    fig.savefig(f'{fig_path}/mean_rna_distance_vs_tcr_hamming_distance_{cancer_type}.svg', format='svg')
