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
load_precalculated_umap_generated = True # set to True if you want to load the precalculated UMAP for generated data
calculate_expansion_prediction = False
skip_tcr_embed = True # take some time, skip to save some time

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
import pickle
import statsmodels.api as sm
import argparse
from torch import nn
import torch.nn.functional as F
fig = plt.figure()
old_fig_size = fig.get_size_inches()

######################################################################
######################################################################
######################################################################

data_path = '/data/che/TRIM/panc'
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

fn_embeddings = f'{data_path}/trained_tcr_embeddings_ae.npz'
with open(fn_embeddings, 'rb') as f:
    npzfile = np.load(f)
    embeddings = npzfile['embeddings']
embeddings = embeddings[mask_patient[mask_null]]

df_preds = df.copy()
df_preds['pseudoclone'] = df_preds['cdr3']
for patient in range(len(dict_patient2num.keys())):
    # with open(f"{data_path}/model/preds_patient{}.npz".format(patient), 'rb') as f:
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

np.random.seed(0)
umapper = umap.UMAP(random_state=0)

args.e_eval_reals_path = '/home/che/TRIM/data_figures/pan-cancer/e_eval_reals.pickle'
args.e_eval_preds_path = '/home/che/TRIM/data_figures/pan-cancer/e_eval_preds.pickle'

if os.path.exists(args.e_eval_reals_path):
    if load_precalculated_umap_real:
        with open(args.e_eval_reals_path, 'rb') as f:
            e = pickle.load(f)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print("Loaded precomputed UMAP for real data from:", args.e_eval_reals_path)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        raise ValueError("Precomputed UMAP for real data not found. Please set load_precalculated_umap_real to True or compute it from scratch.")
else:
    e = umapper.fit_transform(df.iloc[:, :npca].values)
    with open(args.e_eval_reals_path, 'wb') as f:
        pickle.dump(e, f)
    print("Computed UMAP for real data and saved to:", args.e_eval_reals_path)

if os.path.exists(args.e_eval_preds_path):
    if load_precalculated_umap_generated:
        with open(args.e_eval_preds_path, 'rb') as f:
            e_pred = pickle.load(f)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print("Loaded precomputed UMAP for generated data from:", args.e_eval_preds_path)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        raise ValueError("Precomputed UMAP for generated data not found. Please set load_precalculated_umap_generated to True or compute it from scratch.")
else:
    e_pred = umapper.transform(preds_rna)
    with open(args.e_eval_preds_path, 'wb') as f:
        pickle.dump(e_pred, f)
    print("Computed UMAP for generated data and saved to:", args.e_eval_preds_path)

mask_to_plot = np.ones(preds_rna.shape[0]).astype(bool)
# mask_to_plot = df['patient'].apply(lambda tmp: tmp in [4, 12, 18])
## real
loc_dict = {ct: i for i, ct in enumerate(sorted(df['loc'].unique()))}
r = np.random.choice(range(e[mask_to_plot].shape[0]), e[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
colors = mpl.cm.tab10(np.linspace(0, 1, 8))[:2]
custom_cmap = mpl.colors.ListedColormap(colors)
# scat = ax.scatter(e[mask_to_plot][r, 0], e[mask_to_plot][r, 1], s=1, cmap=mpl.cm.viridis, c=df['loc'][mask_to_plot][r].apply(lambda tmp: loc_dict[tmp]))
scat = ax.scatter(e[mask_to_plot][r, 0], e[mask_to_plot][r, 1], s=1, cmap=custom_cmap, c=df['loc'][mask_to_plot][r].apply(lambda tmp: loc_dict[tmp]))
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
# ax.legend(scat.legend_elements()[0], loc_dict.keys(), loc="upper left")
lims = [ax.get_xlim(), ax.get_ylim()]
fig.savefig(f'{fig_path}/real_data_by_normal_tumor.png', dpi=300)
fig.savefig(f'{fig_path}/real_data_by_normal_tumor.svg', format='svg')


ct_dict = {ct: i for i, ct in enumerate(sorted(df['cancerType'].unique()))}
r = np.random.choice(range(e[mask_to_plot].shape[0]), e[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
colors = mpl.cm.tab10(np.linspace(0, 1, 8))[5:8]
custom_cmap = mpl.colors.ListedColormap(colors)
# scat = ax.scatter(e[mask_to_plot][r, 0], e[mask_to_plot][r, 1], s=1, cmap=mpl.cm.viridis, c=df['cancerType'][mask_to_plot][r].apply(lambda tmp: ct_dict[tmp]))
scat = ax.scatter(e[mask_to_plot][r, 0], e[mask_to_plot][r, 1], s=1, cmap=custom_cmap, c=df['cancerType'][mask_to_plot][r].apply(lambda tmp: ct_dict[tmp]))
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
# ax.legend(scat.legend_elements()[0], ct_dict.keys(), loc="upper left", fontsize=6)
[ax.set_xlim(lims[0]), ax.set_ylim(lims[1])]
fig.savefig(f'{fig_path}/real_data_by_cancer_type.png')
fig.savefig(f'{fig_path}/real_data_by_cancer_type.svg', format='svg')


r = np.random.choice(range(e[mask_to_plot].shape[0]), e[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
ax.scatter(e[mask_to_plot][r, 0], e[mask_to_plot][r, 1], s=1, cmap=mpl.cm.viridis, c=df['cdr3_count'][mask_to_plot][r], vmax=10)
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
[ax.set_xlim(lims[0]), ax.set_ylim(lims[1])]
fig.savefig(f'{fig_path}/real_data_by_clone.png')
fig.savefig(f'{fig_path}/real_data_by_clone.svg', format='svg')

cellt_dict = {ct: i for i, ct in enumerate(sorted(df['cellType'].unique()))}
r = np.random.choice(range(e[mask_to_plot].shape[0]), e[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
colors = np.where(df['cellType'][mask_to_plot][r] == 'CD4', '#F781BF', '#999999')
scat = ax.scatter(e[mask_to_plot][r, 0], e[mask_to_plot][r, 1], s=1, c=colors)
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
[ax.set_xlim(lims[0]), ax.set_ylim(lims[1])]
fig.savefig(f'{fig_path}/real_data_by_cell_type.png')
fig.savefig(f'{fig_path}/real_data_by_cell_type.svg', format='svg')

## generated
r = np.random.choice(range(e_pred[mask_to_plot].shape[0]), e_pred[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
colors = mpl.cm.tab10(np.linspace(0, 1, 8))[:2]
custom_cmap = mpl.colors.ListedColormap(colors)
scat = ax.scatter(e_pred[mask_to_plot][r, 0], e_pred[mask_to_plot][r, 1], s=1, cmap=custom_cmap, c=df['loc'][mask_to_plot][r].apply(lambda tmp: loc_dict[tmp]))
# scat = ax.scatter(e_pred[mask_to_plot][r, 0], e_pred[mask_to_plot][r, 1], s=1, cmap=mpl.cm.viridis, c=df['loc'][mask_to_plot][r].apply(lambda tmp: loc_dict[tmp]))
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
# ax.legend(scat.legend_elements()[0], loc_dict.keys(), loc="upper left")
[ax.set_xlim(lims[0]), ax.set_ylim(lims[1])]
fig.savefig(f'{fig_path}/generate_data_by_normal_tumor.png')
fig.savefig(f'{fig_path}/generate_data_by_normal_tumor.svg', format='svg')


r = np.random.choice(range(e_pred[mask_to_plot].shape[0]), e_pred[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
colors = mpl.cm.tab10(np.linspace(0, 1, 8))[5:8]
custom_cmap = mpl.colors.ListedColormap(colors)
scat = ax.scatter(e_pred[mask_to_plot][r, 0], e_pred[mask_to_plot][r, 1], s=1, cmap=custom_cmap, c=df['cancerType'][mask_to_plot][r].apply(lambda tmp: ct_dict[tmp]))
# scat = ax.scatter(e_pred[mask_to_plot][r, 0], e_pred[mask_to_plot][r, 1], s=1, cmap=mpl.cm.viridis, c=df['cancerType'][mask_to_plot][r].apply(lambda tmp: ct_dict[tmp]))
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
# ax.legend(scat.legend_elements()[0], ct_dict.keys(), loc="upper left", fontsize=6)
[ax.set_xlim(lims[0]), ax.set_ylim(lims[1])]
fig.savefig(f'{fig_path}/generate_data_by_cancer_type.png')
fig.savefig(f'{fig_path}/generate_data_by_cancer_type.svg', format='svg')


r = np.random.choice(range(e_pred[mask_to_plot].shape[0]), e_pred[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
ax.scatter(e_pred[mask_to_plot][r, 0], e_pred[mask_to_plot][r, 1], s=1, cmap=mpl.cm.viridis, c=pseudo_tcr_counts[mask_to_plot][r], vmax=10)
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
[ax.set_xlim(lims[0]), ax.set_ylim(lims[1])]
fig.savefig(f'{fig_path}/generate_data_by_clone.png')
fig.savefig(f'{fig_path}/generate_data_by_clone.svg', format='svg')


# CellType annotations (CD4, CD8 for generated data) ---------------------
# 1. train a SVM in real data
svm = SVC(kernel='linear', random_state=0)
train_X = df.iloc[:, :npca].values
train_Y = df['cellType'].values
svm.fit(train_X, train_Y)
y_pred = svm.predict(preds_rna)
# 2. plot the results using predicted cellType labels
r = np.random.choice(range(e_pred[mask_to_plot].shape[0]), e_pred[mask_to_plot].shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
colors = np.where(y_pred[mask_to_plot][r] == 'CD4', '#F781BF', '#999999')
scat = ax.scatter(e_pred[mask_to_plot][r, 0], e_pred[mask_to_plot][r, 1], s=1, c=colors)
[ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
# ax.legend(scat.legend_elements()[0], ct_dict.keys(), loc="upper left", fontsize=6)
[ax.set_xlim(lims[0]), ax.set_ylim(lims[1])]
fig.savefig(f'{fig_path}/generate_data_by_cell_type.png')
fig.savefig(f'{fig_path}/generate_data_by_cell_type.svg', format='svg')


fig.clf()
ax = fig.subplots(1, 1)
colors = mpl.cm.tab10(np.linspace(0, 1, 8))[5:8]
custom_cmap = mpl.colors.ListedColormap(colors)
make_legend(ax, ['ESCA', 'THCA', 'UCEC'], cmap=custom_cmap)
fig.savefig(f'{fig_path}/cancer_type_legend.png', dpi=300)
fig.savefig(f'{fig_path}/cancer_type_legend.svg', format='svg')

fig.clf()
ax = fig.subplots(1, 1)
colors = mpl.cm.tab10(np.linspace(0, 1, 8))[:2]
custom_cmap = mpl.colors.ListedColormap(colors)
make_legend(ax, ['Normal Tissue', 'Tumor'], cmap=custom_cmap)
fig.savefig(f'{fig_path}/normal_tumor_legend.png', dpi=300)
fig.savefig(f'{fig_path}/normal_tumor_legend.svg', format='svg')

# fig.clf()
ax = fig.subplots(1, 1)
# make_legend(ax, ['CD4', 'CD8'], cmap=mpl.cm.Paired)
make_legend(ax, ['CD4', 'CD8'], cmap=mpl.colors.ListedColormap(['#F781BF', '#999999']))
fig.savefig(f'{fig_path}/cell_type_legend.png')
fig.savefig(f'{fig_path}/cell_type_legend.svg', format='svg')



#######
# for clonotype diversity/count analysis in cluster subsets

# I don't think we need this mask? As preds_rna is already generated by leave-one-out format
# mask = ~mask_train

nclust = 20
km = sklearn.cluster.MiniBatchKMeans(nclust, random_state=0)
clusts_real = km.fit_predict(df.iloc[:, :npca].values) # clusts_real = km.fit_predict(df[mask].iloc[:, :npca].values)
clusts_pred = km.predict(preds_rna) # clusts_pred = km.predict(preds_rna[mask])

means = []
for clust in range(nclust):
    a = np.log(df['cdr3_count'][clusts_real == clust].mean()) #a = np.log(df['cdr3_count'][mask][clusts_real == clust].mean())
    b = np.log(pseudo_tcr_counts[clusts_pred == clust].mean()) # b = np.log(pseudo_tcr_counts[mask][clusts_pred == clust].mean())
    means.append([a, b])
means = np.array(means)

model_results = sm.OLS(means[:, 1], sm.add_constant(means[:, 0])).fit()
print('r = {:.2f}'.format(np.corrcoef(means[:, 0], means[:, 1])[0, 1]))
fig.clf()
ax = fig.subplots(1, 1)
ax.scatter(means[:, 0], means[:, 1])
[ax.set_xlabel('Real avg clonality'), ax.set_ylabel('Predicted avg clonality')]
# [ax.set_xlim([0, 1]), ax.set_ylim([0, 1])]
ax.plot(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], .01), model_results.params[0] + model_results.params[1] * np.arange(ax.get_xlim()[0], ax.get_xlim()[1], .01), c='k', linestyle='--', label='r = {:0.3f}'.format(np.sign( model_results.params[1]) * np.sqrt(model_results.rsquared)))
ax.legend(loc='lower right')
fig.savefig(f'{fig_path}/clone_by_cluster_mean.png')
fig.savefig(f'{fig_path}/clone_by_cluster_mean.svg', format='svg')



means = []
for clust in range(nclust):
    a = np.unique(df['cdr3'][clusts_real == clust]).shape[0] / (clusts_real == clust).sum() # a = np.unique(df['cdr3'][mask][clusts_real == clust]).shape[0] / (clusts_real == clust).sum()
    b = np.unique(pseudo_tcrs[clusts_pred == clust]).shape[0] / (clusts_pred == clust).sum() # b = np.unique(pseudo_tcrs[mask][clusts_pred == clust]).shape[0] / (clusts_pred == clust).sum()
    means.append([a, b])
means = np.array(means)

model_results = sm.OLS(means[:, 1], sm.add_constant(means[:, 0])).fit()
print('r = {:.2f}'.format(np.corrcoef(means[:, 0], means[:, 1])[0, 1]))
fig.clf()
ax = fig.subplots(1, 1)
ax.scatter(means[:, 0], means[:, 1])
[ax.set_xlabel('Real clonal diversity'), ax.set_ylabel('Predicted clonal diversity')]
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
[ax.set_xlim(lims), ax.set_ylim(lims)]
ax.plot(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], .01), model_results.params[0] + model_results.params[1] * np.arange(ax.get_xlim()[0], ax.get_xlim()[1], .01), c='k', linestyle='--', label='r = {:0.3f}'.format(np.sign( model_results.params[1]) * np.sqrt(model_results.rsquared)))
ax.legend(loc='lower right')
fig.savefig(f'{fig_path}/clone_by_cluster.png')
fig.savefig(f'{fig_path}/clone_by_cluster.svg', format='svg')


######################################################################
# to make hamming vs rna charts
# skip, see /home/che/TRIM/git/tcr/analysis/pan-cancer/0.1.eval_rna_pairwise_dist.py
#######################################################################

if not skip_tcr_embed:
    # correlation between rna and our cnn embeddings
    r = np.random.choice(range(df.shape[0]), 10000, replace=False)
    dists1 = sklearn.metrics.pairwise_distances(embeddings[r], embeddings[r])
    dists2 = sklearn.metrics.pairwise_distances(df.iloc[r, :npca], df.iloc[r, :npca])
    mask_triu = (np.triu(np.ones([dists1.shape[0], dists1.shape[0]])) - np.eye(dists1.shape[0])).flatten() == 1
    mask_triu = np.logical_and(mask_triu, np.logical_and(dists1.flatten() > 1e-6, dists2.flatten() > 1e-6)) #b/c we are running this on cells, not TCRs
    dists1 = dists1.flatten()[mask_triu]
    dists2 = dists2.flatten()[mask_triu]
    corr = np.corrcoef(dists1, dists2)[0, 1]
    print("r = {:.3f}".format(corr))
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.scatter(dists1, dists2, s=1, alpha=.5)
    model_results = sm.OLS(dists2, sm.add_constant(dists1)).fit()
    print(np.sqrt(model_results.rsquared))
    xlim = ax.get_xlim()
    ax.plot(np.arange(xlim[0], xlim[1], .01), model_results.params[0] + model_results.params[1] * np.arange(xlim[0], xlim[1], .01), c='k', linestyle='--', label='r = {:0.3f}'.format(np.sign( model_results.params[1]) * np.sqrt(model_results.rsquared)))
    ax.legend(loc='lower right')
    fig.savefig(f'{fig_path}/rna_tcr_cnn_embeddings.png')
    fig.savefig(f'{fig_path}/rna_tcr_cnn_embeddings.svg', format='svg')

    # correlation between rna and our tcr-bert embeddings
    from transformers import BertModel
    from transformers import AutoModel
    from transformers import FeatureExtractionPipeline
    sys.path.append('/home/che/TRIM/tcr-bert/tcr')
    import featurization as ft
    tcrbert_model = BertModel.from_pretrained("wukevin/tcr-bert")

    def load_embed_pipeline(model_dir: str, device: int):
        """
        Load the pipeline object that gets embeddings
        """
        model = AutoModel.from_pretrained(model_dir)
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
        pipeline = FeatureExtractionPipeline(model, tok, device=device)
        return pipeline


    if os.path.exists(f'/home/che/TRIM/data_figures/pan-cancer/tcrbert_embeddings.pkl'):
        with open(f'/home/che/TRIM/data_figures/pan-cancer/tcrbert_embeddings.pkl', 'rb') as f:
            tcrbert_embeddings = pickle.load(f)
            print('Loaded precomputed TCR-BERT embeddings from disk.')
    else:
        tcrbert_trb_embedder = load_embed_pipeline("wukevin/tcr-bert", device=0)

        s = [' '.join([aa for aa in s.replace(' ', '')]) for s in df['cdr3'].values]
        tcrbert_out = tcrbert_trb_embedder(s)

        # or mean? or pool?
        tcrbert_embeddings = np.stack([np.array(o[0])[0] for o in tcrbert_out])
        # # tcrbert_embeddings = np.stack([np.mean(np.array(o[0]), axis=0) for o in tcrbert_out])
        # # tcrbert_embeddings = np.stack([np.mean(np.array(o[0])[1:-1], axis=0) for o in tcrbert_out])

        # Save the TCR-BERT embeddings to disk for future use
        with open(f'/home/che/TRIM/data_figures/pan-cancer/tcrbert_embeddings.pkl', 'wb') as f:
            pickle.dump(tcrbert_embeddings, f)
        print('Computed TCR-BERT embeddings and saved to disk at /home/che/TRIM/data_figures/pan-cancer/tcrbert_embeddings.pkl')

    r = np.random.choice(range(df.shape[0]), 10000, replace=False)
    dists1 = sklearn.metrics.pairwise_distances(tcrbert_embeddings[r], tcrbert_embeddings[r])
    dists2 = sklearn.metrics.pairwise_distances(df.iloc[r, :npca], df.iloc[r, :npca])
    mask_triu = (np.triu(np.ones([dists1.shape[0], dists1.shape[0]])) - np.eye(dists1.shape[0])).flatten() == 1
    mask_triu = np.logical_and(mask_triu, np.logical_and(dists1.flatten() > 1e-4, dists2.flatten() > 1e-4)) #b/c we are running this on cells, not TCRs
    dists1 = dists1.flatten()[mask_triu]
    dists2 = dists2.flatten()[mask_triu]
    corr = np.corrcoef(dists1, dists2)[0, 1]
    print("r = {:.3f}".format(corr))
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.scatter(dists1, dists2, s=1, alpha=.5)
    model_results = sm.OLS(dists2, sm.add_constant(dists1)).fit()
    print(np.sqrt(model_results.rsquared))
    xlim = ax.get_xlim()
    ax.plot(np.arange(xlim[0], xlim[1], .01), model_results.params[0] + model_results.params[1] * np.arange(xlim[0], xlim[1], .01), c='k', linestyle='--', label='r = {:0.3f}'.format(np.sign( model_results.params[1]) * np.sqrt(model_results.rsquared)))
    ax.legend(loc='lower right')
    fig.savefig(f'{fig_path}/rna_tcr_bert_embeddings.png')
    fig.savefig(f'{fig_path}/rna_tcr_bert_embeddings.svg', format='svg')
else:
    print('Skipping TCR-BERT or TCR CNN embeddings analysis...')

######################################################################
######################################################################
######################################################################
# For expansion prediction
######################################################################
######################################################################
######################################################################

print('Starting expansion prediction analysis...')
class Loader(object):
    """A Loader class for feeding numpy matrices into tensorflow models."""

    def __init__(self, data, shuffle=False):
        """Initialize the loader with data and optionally with labels."""
        self.start = 0
        self.epoch = 0
        if type(data) == list:
            self.data = data
        else:
            self.data = [data]

        if shuffle:
            self.r = list(range(data[0].shape[0]))
            np.random.shuffle(self.r)
            self.data = [x[self.r] for x in self.data]

    def next_batch(self, batch_size=100):
        """Yield just the next batch."""
        num_rows = self.data[0].shape[0]

        if self.start + batch_size < num_rows:
            batch = [x[self.start:self.start + batch_size] for x in self.data]
            self.start += batch_size
        else:
            self.epoch += 1
            batch_part1 = [x[self.start:] for x in self.data]
            batch_part2 = [x[:batch_size - (x.shape[0] - self.start)] for x in self.data]
            batch = [np.concatenate([x1, x2], axis=0) for x1, x2 in zip(batch_part1, batch_part2)]

            self.start = batch_size - (num_rows - self.start)

        if len(self.data) == 1:
            return batch[0] # don't return length-1 list
        else:
            return batch # return list of data matrixes

    def iter_batches(self, batch_size=100):
        """Iterate over the entire dataset in batches."""
        num_rows = self.data[0].shape[0]

        end = 0

        if batch_size > num_rows:
            if len(self.data) == 1:
                yield [x for x in self.data][0]
            else:
                yield [x for x in self.data]
        else:
            for i in range(num_rows // batch_size):
                start = i * batch_size
                end = (i + 1) * batch_size

                if len(self.data) == 1:
                    yield [x[start:end] for x in self.data][0]
                else:
                    yield [x[start:end] for x in self.data]
            if end < num_rows:
                if len(self.data) == 1:
                    yield [x[end:] for x in self.data][0]
                else:
                    yield [x[end:] for x in self.data]

class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nbase = self.args.n_channels_base

        n_conditions = 3 * args.dim_state_embedding

        # RNA
        total_dim_in_rna = args.dimrna + n_conditions
        self.encoder_rna = MLP(dim_in=total_dim_in_rna, dim_out=args.dimz * 2, nbase=nbase * 1)
        self.decoder_rna = MLP(dim_in=args.dimz + n_conditions, dim_out=args.dimrna, nbase=nbase * 1, decoder=True)

        # TCR
        total_dim_in_tcr = args.dimtcr + n_conditions
        self.encoder_tcr = MLP(dim_in=total_dim_in_tcr, dim_out=args.dimz * 2, nbase=nbase * 1)
        self.decoder_tcr = MLP(dim_in=args.dimz + n_conditions, dim_out=args.dimtcr, nbase=nbase * 1, decoder=True)

        # embeddings
        self.register_buffer('bloodtumor_embeddings_matrix', torch.zeros(2, args.dim_state_embedding))
        self.register_buffer('cancertype_embeddings_matrix', torch.zeros(3, args.dim_state_embedding))
        self.register_buffer('patient_embeddings_matrix', torch.zeros(args.num_patients, args.dim_state_embedding))

        # self.bloodtumor_embeddings_matrix = torch.nn.Embedding(2, args.dim_state_embedding)
        # self.prepost_embeddings_matrix = torch.nn.Embedding(2, args.dim_state_embedding)
        # self.patient_embeddings_matrix = torch.nn.Embedding(args.num_patients, args.dim_state_embedding)

        # ops
        self.lrelu = torch.nn.LeakyReLU()


        # learnable patient embeddings
        self.mlp_patient_embeddings_rna = MLP(dim_in=args.dimrna, dim_out=args.dim_state_embedding, nbase=nbase * 1)
        self.mlp_patient_embeddings_tcr = MLP(dim_in=args.dimtcr, dim_out=args.dim_state_embedding, nbase=nbase * 1)

    def forward(self, x, embeddings):
        x_rna = torch.cat([x[0]] + embeddings, axis=-1)
        x_tcr = torch.cat([x[1]] + embeddings, axis=-1)

        z_rna = self.encoder_rna(x_rna)
        z_tcr = self.encoder_tcr(x_tcr)

        mu = torch.mean(torch.stack([z_rna[:, :self.args.dimz], z_tcr[:, :self.args.dimz]]), 0)
        logvar = torch.mean(torch.stack([z_rna[:, self.args.dimz:], z_tcr[:, self.args.dimz:]]), 0)

        z = reparameterize(mu, logvar)

        shared_layer = torch.cat([z] + embeddings, axis=-1)

        recon_tcr = self.decoder_tcr(shared_layer)
        recon_rna = self.decoder_rna(shared_layer)

        return recon_rna, recon_tcr, [mu, logvar, [z_rna, z_tcr]]

    def sample(self, z, embeddings, mu_logvar=None):
        if mu_logvar is not None:
            z = reparameterize(mu_logvar[0], mu_logvar[1])

        shared_layer = torch.cat([z] + embeddings, axis=-1)

        recon_tcr = self.decoder_tcr(shared_layer)
        recon_rna = self.decoder_rna(shared_layer)

        return recon_rna, recon_tcr, [_, _, shared_layer]

    def map_to_patient_embeddings(self, x, labels, num_labels):
        e = self.mlp_patient_embeddings(x)

        e = self.groupby_mean(e, labels, num_labels)

        return e

    def groupby_mean(self, value, labels, num_labels):
        for i_pid in range(num_labels):
            if (labels == i_pid).sum() == 0:
                value = torch.cat([value, torch.zeros_like(value[0])[np.newaxis, :]], axis=0)
                labels = torch.cat([labels, torch.Tensor([i_pid]).to(device)], axis=0)

        uniques = labels.unique().tolist()
        labels = labels.tolist()

        key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
        val_key = {int(val): int(key) for key, val in zip(uniques, range(len(uniques)))}
        
        labels = torch.LongTensor(list(map(key_val.get, labels))).to(device)
        
        labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))
        
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

        result = torch.zeros_like(unique_labels, dtype=torch.float).to(device)

        result = result.scatter_add_(0, labels, value)
        result = result / labels_count.float().unsqueeze(1)

        new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].type(torch.int32).tolist())))

        return result, new_labels

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        nbase = kwargs['nbase']
        dim_in = kwargs["dim_in"]
        dim_out = kwargs["dim_out"]
        layers = [1, 2, 4]
        if "decoder" in kwargs:
            layers = layers[::-1]

        self.layer1 = nn.Linear(in_features=dim_in, out_features=nbase // layers[0])
        self.layer2 = nn.Linear(in_features=nbase // layers[0], out_features=nbase // layers[1])
        self.layer3 = nn.Linear(in_features=nbase // layers[1], out_features=nbase // layers[2])
        self.out = nn.Linear(in_features=nbase // layers[2], out_features=dim_out)

        self.bn1 = nn.BatchNorm1d(nbase // layers[0])
        self.bn2 = nn.BatchNorm1d(nbase // layers[1])
        self.bn3 = nn.BatchNorm1d(nbase // layers[2])

        if 'act' not in kwargs:
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = kwargs['act']

    def forward(self, x):
        h1 = self.act(self.bn1(self.layer1(x)))
        h2 = self.act(self.bn2(self.layer2(h1)))
        h3 = self.act(self.bn3(self.layer3(h2)))
        out = self.out(h3)

        return out

def numpy2torch(x, type=torch.FloatTensor):

    return torch.from_numpy(x).type(type).to(device)

def reparameterize(mu, logvar, clamp=5):
    std = logvar.mul(0.5).exp_()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)




args_folder = '/data/che/TRIM/panc/model/args.txt'
args = argparse.ArgumentParser().parse_args()
with open(args_folder, 'r') as f:
    args.__dict__ = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator(args)
G.load_state_dict(torch.load('/data/che/TRIM/panc/model/model.pth'))

G.eval()
G = G.to(device)




args_folder = '/data/che/TRIM/panc/model/args.txt'
args = argparse.ArgumentParser().parse_args()
with open(args_folder, 'r') as f:
    args.__dict__ = json.load(f)


npca = 100
with open('/data/che/TRIM/panc/data_pca.npz', 'rb') as f:
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


fn_embeddings = '/data/che/TRIM/panc/trained_tcr_embeddings_ae.npz'
with open(fn_embeddings, 'rb') as f:
    npzfile = np.load(f)
    embeddings = npzfile['embeddings']
embeddings = embeddings[mask_patient[mask_null]]


dict_loc2num = {'P': 0, 'N': 0, 'T': 1,}
dict_patient2num = {p: i for i, p in enumerate(sorted(df['patient'].unique()))}
dict_cancertype2num = {c: i for i, c in enumerate(sorted(df['cancerType'].unique()))}
df['loc'] = df['loc'].apply(lambda tmp: dict_loc2num[tmp])
df['patient'] = df['patient'].apply(lambda tmp: dict_patient2num[tmp])
df['cancerType'] = df['cancerType'].apply(lambda tmp: dict_cancertype2num[tmp])
col_loc = df.columns[npca:].tolist().index('loc')
col_patient = df.columns[npca:].tolist().index('patient')
col_cancertype = df.columns[npca:].tolist().index('cancerType')


print('Starting to load output...')
t = time.time()

with open('/data/che/TRIM/panc/model/preds.npz', 'rb') as f:
    npzfile = np.load(f)
    preds_rna = npzfile['preds_rna']
    preds_tcr = npzfile['preds_tcr']
    recon_rna_z = npzfile['recon_rna_z']
    recon_tcr_z = npzfile['recon_tcr_z']

    pseudo_tcrs = npzfile['pseudo_tcrs']
    tcr_dists = npzfile['tcr_dists']
    thresh_fitted = npzfile['thresh_fitted']

    mask_train = npzfile['mask_train']

print('Loaded output in {:.1f} s'.format(time.time() - t))



mask_to_plot = np.logical_and(df['patient'].apply(lambda tmp: tmp in [4, 12, 18]), df['loc'] == 0)[mask]

recon_rna_z_torch = numpy2torch(recon_rna_z[mask_to_plot])
recon_tcr_z_torch = numpy2torch(recon_tcr_z[mask_to_plot])


mu = torch.stack([recon_rna_z_torch[:, :args.dimz], recon_tcr_z_torch[:, :args.dimz]], axis=0).mean(axis=0)
logvar = torch.stack([recon_rna_z_torch[:, args.dimz:], recon_tcr_z_torch[:, args.dimz:]], axis=0).mean(axis=0)


########
if calculate_expansion_prediction:

    num_points = 20000
    num_samples = int(num_points / mu.shape[0])
    print('num_samples: {}'.format(num_samples))


    all_out_rna = []
    all_out_tcr = []

    for _ in range(num_samples):
        out_rna_ = []
        out_tcr_ = []
        batch_random = reparameterize(mu, logvar)

        ones = torch.ones(batch_random.shape[0]).to(device).type(torch.int32)

        for bloodtumor in [0, 1]:
            batch_cancertype_embeddings = torch.index_select(G.cancertype_embeddings_matrix, 0, numpy2torch(df['cancerType'][mask_to_plot].values).type(torch.int32) * ones)
            batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, bloodtumor * ones)
            batch_patient_embeddings  = torch.index_select(G.patient_embeddings_matrix, 0, numpy2torch(df['patient'][mask_to_plot].values).type(torch.int32))

            out_rna, out_tcr, [_, _, _] = G.sample(z=batch_random,
                                                embeddings=[batch_cancertype_embeddings, batch_bloodtumor_embeddings, batch_patient_embeddings])
            out_rna_.append(out_rna.detach().cpu().numpy())
            out_tcr_.append(out_tcr.detach().cpu().numpy())
        all_out_rna.append(out_rna_)
        all_out_tcr.append(out_tcr_)

    all_out_rna = np.concatenate(all_out_rna, axis=1)
    all_out_tcr = np.concatenate(all_out_tcr, axis=1)
    print('Got predictions')

    # get pseudoclones per condition
    all_dists = []
    for condition in range(2):

        dists = sklearn.metrics.pairwise_distances(all_out_tcr[condition], all_out_tcr[condition], metric='l1')
        all_dists.append(dists)
        print('dists {} done'.format(condition))


    pids_repeated = np.tile(df['patient'][mask_to_plot], num_samples)
    all_pseudo_tcrs = []
    for condition in range(2):

        dists = all_dists[condition]

        thresh = thresh_fitted / 10
        pseudo_tcrs = - 10 * np.ones(all_out_tcr[condition].shape[0])
        curr_tcr_id = 0
        while (pseudo_tcrs == -10).sum() > 0:
            if curr_tcr_id % 1000 == 0:
                print("{:>5}: {:>5}".format(curr_tcr_id, pseudo_tcrs.shape[0] - (pseudo_tcrs != -10).sum()))
            i = np.random.choice(np.argwhere(pseudo_tcrs == -10).flatten())

            row_dists = dists[i]

            mask = np.logical_and(row_dists < thresh, pseudo_tcrs == -10)
            mask = np.logical_and(mask, pids_repeated[i] == pids_repeated)
            pseudo_tcrs[mask] = curr_tcr_id

            curr_tcr_id += 1

        all_pseudo_tcrs.append(pseudo_tcrs)

    all_pseudo_tcrs = np.stack(all_pseudo_tcrs)

    print(["{:.3f}".format(len(np.unique(t)) / all_pseudo_tcrs.shape[1]) for t in all_pseudo_tcrs])


    all_pseudo_clonalities = []
    for i in range(all_out_tcr.shape[1]):
        clonalities = []
        for condition in range(2):
            pseudo_id = all_pseudo_tcrs[condition, i]
            clonality = (all_pseudo_tcrs[condition] == pseudo_id).sum()
            clonalities.append(clonality)
        all_pseudo_clonalities.append(clonalities)

    all_pseudo_clonalities = np.array(all_pseudo_clonalities)
    all_pseudo_clonalities = np.mean(np.array(np.array_split(all_pseudo_clonalities, num_samples)), 0)


    all_real_clonalities = np.zeros(all_pseudo_clonalities.shape)
    tcr_counts_by_loc = df[['cdr3', 'loc']].value_counts()
    for i, cdr in enumerate(df['cdr3'][mask_to_plot]):
        if cdr not in tcr_counts_by_loc[tcr_counts_by_loc.index.get_level_values(1) == 0].index:
            all_real_clonalities[i, 0] == 0
        else:
            all_real_clonalities[i, 0] = tcr_counts_by_loc[tcr_counts_by_loc.index.get_level_values(1) == 0].loc[cdr].values[0]

        if cdr not in tcr_counts_by_loc[tcr_counts_by_loc.index.get_level_values(1) == 1].index:
            all_real_clonalities[i, 1] == 0
        else:
            all_real_clonalities[i, 1] = tcr_counts_by_loc[tcr_counts_by_loc.index.get_level_values(1) == 1].loc[cdr].values[0]


    # save all_real_clonalities, all_pseudo_clonalities into npy files
    np.save('/home/che/TRIM/data_figures/pan-cancer/expansion_all_out_tcr.npy', all_out_tcr)
    np.save('/home/che/TRIM/data_figures/pan-cancer/expansion_all_out_rna.npy', all_out_rna)
    np.save('/home/che/TRIM/data_figures/pan-cancer/all_real_clonalities.npy', all_real_clonalities)
    np.save('/home/che/TRIM/data_figures/pan-cancer/all_pseudo_clonalities.npy', all_pseudo_clonalities)
else:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Loaded expansion prediction results')
    print('if new data is generated, need to re-run')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('-'*80)
    all_real_clonalities = np.load('/home/che/TRIM/data_figures/pan-cancer/all_real_clonalities.npy')
    all_pseudo_clonalities = np.load('/home/che/TRIM/data_figures/pan-cancer/all_pseudo_clonalities.npy')


x = all_real_clonalities[:, 1] > all_real_clonalities[:, 0]
y = all_pseudo_clonalities[:, 1] / all_pseudo_clonalities[:, 0]
y *= (all_pseudo_clonalities[:, 1] > 1).astype(int)
# y = y * (y < 1.5).astype(int)

fpr, tpr, thresholds = sklearn.metrics.roc_curve(x, y, pos_label=1)
roc_auc = sklearn.metrics.auc(fpr, tpr)
print(roc_auc)

fig.clf()
ax = fig.subplots(1, 1)
ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
[ax.set_xlim([0.0, 1.0]), ax.set_ylim([0.0, 1.05])]
[ax.set_xlabel("False Positive Rate"), ax.set_ylabel("True Positive Rate")]
ax.legend(loc="lower right")
fig.savefig(f'{fig_path}/expansion_prediction_trim.png')
fig.savefig(f'{fig_path}/expansion_prediction_trim.svg', format='svg')



# umapper = umap.UMAP()
# e = umapper.fit_transform(all_out_rna[0, :6226, :])

# e_pred = umapper.transform(all_out_rna[0, :6226, :])


## real
# r = np.random.choice(range(e_pred.shape[0]), e_pred.shape[0], replace=False)
# fig.clf()
# ax = fig.subplots(1, 1)
# scat = ax.scatter(e_pred[r, 0], e_pred[r, 1], s=10, cmap=mpl.cm.viridis, c=mask_tmp[r])
# [ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
# ax.legend(scat.legend_elements()[0], loc_dict.keys(), loc="upper left")
# fig.savefig(f{fig_path}/tmp.png')


######################################################################
# baseline expansion prediction
npca = 100
with open('/data/che/TRIM/panc/data_pca.npz', 'rb') as f:
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
df['loc'] = df['loc'].apply(lambda tmp: dict_loc2num[tmp])
df['patient'] = df['patient'].apply(lambda tmp: dict_patient2num[tmp])
col_loc = df.columns[npca:].tolist().index('loc')
col_patient = df.columns[npca:].tolist().index('patient')


def tmp(series, query):
    if not query in series:
        return 0
    else:
        return series.loc[query]

tcr_counts_by_loc = df[['cdr3', 'loc']].value_counts()
df['cdr3_count_N'] = np.array([tmp(tcr_counts_by_loc.loc[tcr], 0) for tcr in df['cdr3'].tolist()])
df['cdr3_count_T'] = np.array([tmp(tcr_counts_by_loc.loc[tcr], 1) for tcr in df['cdr3'].tolist()])
expanded = (df['cdr3_count_N'] < df['cdr3_count_T']).astype(int)


def get_drop_duplicates_mask(x, numpy=True):
    mask = []
    uniques = set()
    for row in x:
        row = row
        if not numpy:
            row = row.detach().cpu().numpy()
        row = tuple(row)
        if row not in uniques:
            mask.append(True)
        else:
            mask.append(False)
        uniques.add(row)
    return mask

def baseline_expansion_prediction(train_on, baseline_model, figname='tmp.png'):
    # train_on         can be  ['rna', 'tcr', 'both']
    # baseline_model   can be  ['svm', 'knn', 'random_forest', 'NN']
    import sklearn.svm
    import sklearn.neighbors
    import sklearn.ensemble
    import sklearn.neural_network


    # mask_train_baseline = np.logical_and((all_real_clonalities > 1).all(axis=1), df['loc'] == 0)
    # mask_train_baseline = mask_train#np.logical_and(mask_train_baseline, mask_train)
    mask_train_baseline = df['patient'].apply(lambda tmp: tmp not in [4, 12, 18])
    # mask_train_baseline = np.logical_and(mask_train_baseline, df['cdr3_count_N'] >= 1)
    # mask_train_baseline = np.logical_and(mask_train_baseline, df['cdr3_count_T'] >= 1)
    x_rna_masked_train = df.iloc[:, :npca].values[mask_train_baseline]
    x_tcr_masked_train = embeddings[mask_train_baseline]
    tumor_expanded_train = expanded[mask_train_baseline]#(all_real_clonalities[mask_train_baseline, 0] < all_real_clonalities[mask_train_baseline, 1]).astype(np.int32)


    # mask_test_baseline = np.logical_and((all_real_clonalities > 1).all(axis=1), df['loc'] == 0)
    # mask_test_baseline = ~mask_train#np.logical_and(mask_test_baseline, ~mask_train)
    mask_test_baseline = np.logical_and(df['patient'].apply(lambda tmp: tmp in [4, 12, 18]), df['loc'] == 0)
    mask_test_baseline = np.logical_and(mask_test_baseline, df['cdr3_count_N'] >= 1)
    mask_test_baseline = np.logical_and(mask_test_baseline, df['cdr3_count_T'] >= 1)
    x_rna_masked_test = df.iloc[:, :npca].values[mask_test_baseline]
    x_tcr_masked_test = embeddings[mask_test_baseline]
    tumor_expanded_test = expanded[mask_test_baseline]#(all_real_clonalities[mask_test_baseline, 0] < all_real_clonalities[mask_test_baseline, 1]).astype(np.int32)



    if train_on == 'rna':
        classifier_x_train = x_rna_masked_train
        classifier_x_test = x_rna_masked_test
    elif train_on == 'tcr':
        classifier_x_train = x_tcr_masked_train
        classifier_x_test = x_tcr_masked_test
    elif train_on == 'both':
        classifier_x_train = np.concatenate([x_rna_masked_train, x_tcr_masked_train], axis=-1)
        classifier_x_test = np.concatenate([x_rna_masked_test, x_tcr_masked_test], axis=-1)
    else:
        raise Exception('bad train_on')



    if baseline_model == 'svm':
        classifier = sklearn.svm.SVC(probability=True, random_state=42)
        r_subsample = np.random.choice(range(mask_train_baseline.sum()), 5000, replace=False)
    elif baseline_model == 'knn':
        classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=20)
        r_subsample = np.random.choice(range(mask_train_baseline.sum()), 20000, replace=False)
    elif baseline_model == 'random_forest':
        classifier = sklearn.ensemble.RandomForestClassifier(random_state=42)
        r_subsample = np.random.choice(range(mask_train_baseline.sum()), 20000, replace=False)
    elif baseline_model == 'NN':
        classifier = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=[400, 200, 100], random_state=42)
        r_subsample = np.random.choice(range(mask_train_baseline.sum()), 20000, replace=False)
    else:
        raise Exception("bad baseline_model")


    mask_drop_dup_train = get_drop_duplicates_mask(x_tcr_masked_train[r_subsample])

    classifier.fit(classifier_x_train[r_subsample][mask_drop_dup_train], tumor_expanded_train[r_subsample][mask_drop_dup_train])
    preds = classifier.predict_proba(classifier_x_test)[:, 1]

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(tumor_expanded_test, preds, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print("ROC: {:.3f}".format(roc_auc))


    fig.clf()
    ax = fig.subplots(1, 1)
    lw = 2
    ax.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    [ax.set_xlim([0.0, 1.0]), ax.set_ylim([0.0, 1.05])]
    [ax.set_xlabel("False Positive Rate"), ax.set_ylabel("True Positive Rate")]
    ax.legend(loc="lower right")
    fig.savefig(figname)
    fig.savefig(figname.replace('.png', '.svg'), format='svg') # also save a svg version

# baseline_expansion_prediction('both', 'knn', f'{fig_path}/expansion_prediction_baseline_knn_both.png')
# baseline_expansion_prediction('both', 'svm', f'{fig_path}/expansion_prediction_baseline_svm_both.png')
# baseline_expansion_prediction('both', 'random_forest', f'{fig_path}/expansion_prediction_baseline_random_forest_both.png')
# baseline_expansion_prediction('both', 'NN', f'{fig_path}/expansion_prediction_baseline_NN_both.png')

# ######################################################################
# ######################################################################
# ######################################################################





