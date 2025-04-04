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
do_fig_4c = False # expansion prediction baseline
do_clont_count_by_cluster = True

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
        raise Exception('Found e_eval_reals.pickle but load_precalculated_umap_real is False. Please delete the file or set load_precalculated_umap_real to True to load it.')
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

args.e_eval_preds_path = '/home/che/TRIM/data_figures/HNSCC/e_eval_preds.pickle'
if os.path.exists(args.e_eval_preds_path):
    if load_precalculated_umap_generated:
        with open(args.e_eval_preds_path, 'rb') as f:
            e_eval_preds = pickle.load(f)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!! Loaded e_eval_preds from', args.e_eval_preds_path)
        print('!!!!!!! if the generated data is updated, please delete the file and re-run to generate a new e_eval_preds.pickle')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        raise Exception('Found e_eval_preds.pickle but load_precalculated_umap_generated is False. Please delete the file or set load_precalculated_umap_generated to True to load it.')
else:
    e_eval_preds = viz_reducer.transform(preds_rna_holdout)
    with open(args.e_eval_preds_path, 'wb') as f:
        pickle.dump(e_eval_preds, f)
    print('Saved e_eval_preds to', args.e_eval_preds_path)

#############################################
#############################################

if calculate_expansion_prediction:
    all_real_clonalities_raw, all_pseudo_clonalities_raw = our_expansion_prediction(output_folders)
    # how accurate is expansion prediction?
    all_real_clonalities = np.concatenate(all_real_clonalities_raw, axis=0)
    all_pseudo_clonalities = np.concatenate(all_pseudo_clonalities_raw, axis=0)

    # save all_real_clonalities, all_pseudo_clonalities into npy files
    # np.save('/home/che/TRIM/data_figures/HNSCC/all_real_clonalities.npy', all_real_clonalities)
    # np.save('/home/che/TRIM/data_figures/HNSCC/all_pseudo_clonalities.npy', all_pseudo_clonalities)
else:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Loaded expansion prediction results')
    print('if new data is generated, need to re-run')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('-'*80)
    all_real_clonalities = np.load('/home/che/TRIM/data_figures/HNSCC/all_real_clonalities.npy')
    all_pseudo_clonalities = np.load('/home/che/TRIM/data_figures/HNSCC/all_pseudo_clonalities.npy')

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
# figure 4a
fig.set_size_inches([4,3])
np.random.seed(0)
a, b = np.unique(pseudo_tcrs_holdout, return_counts=True)
srt = a.argsort()
a = a[srt]
b = b[srt]
clone_count_by_cell_pred = np.take(b, pseudo_tcrs_holdout.astype(np.int32))

r = np.random.choice(range(e_eval_preds.shape[0]), e_eval_preds.shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
clone_count_by_cell_pred = clone_count_by_cell_pred / clone_count_by_cell_pred.sum() * 1000000
ax.scatter(e_eval_preds[r, 0], e_eval_preds[r, 1], s=1, cmap=mpl.cm.viridis, c=np.log(clone_count_by_cell_pred[r]), vmin=0, vmax=2.3)
[ax.set_xticks([]), ax.set_yticks([])]
[ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
fig.savefig(f'{figure_path}/figure4/4a_rna_umap_by_pseudo_clonecount.png')
fig.savefig(f'{figure_path}/figure4/4a_rna_umap_by_pseudo_clonecount.svg', format='svg')

# extended figure 5a 
# rna umap by real and generated, by timepoint and pre/post treatment)
def make_legend_temp(ax, labels, s=20, cmap=mpl.cm.jet, **kwargs):
    numlabs = len(labels)
    for i, label in enumerate(labels):
        if numlabs > 1:
            ax.scatter(0, 0, s=s, c=[cmap(1 * i / (numlabs - 1))], label=label)
        else:
            ax.scatter(0, 0, s=s, c=[cmap(1.)], label=label)
    ax.scatter(0, 0, s=2 * s, c='w')
    ax.legend(**kwargs)

def scatter_helper(embedding1, embedding2, ax, **kwargs):  
    e = np.concatenate([embedding1, embedding2], axis=0)
    l = np.concatenate([np.zeros(embedding1.shape[0]), np.ones(embedding2.shape[0])], axis=0)

    r = np.random.choice(range(e.shape[0]), e.shape[0], replace=False)
    
    if e.shape[0] > 0:
        ax.scatter(e[r, 0], e[r, 1], c=l[r], s=1, **kwargs)

mask1 = np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0)
mask2 = np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 1)
mask3 = np.logical_and(x_label[:, col_bloodtumor] == 1, x_label[:, col_prepost] == 0)
mask4 = np.logical_and(x_label[:, col_bloodtumor] == 1, x_label[:, col_prepost] == 1)

conditions = [
    (mask1, 'Blood Pre-Treatment'),
    (mask2, 'Blood Post-Treatment'),
    (mask3, 'Tumor Pre-Treatment'),
    (mask4, 'Tumor Post-Treatment')
]

for idx, (mask, title) in enumerate(conditions):
    fig.clf()
    fig.set_size_inches([3.5, 3])
    ax = fig.subplots(1, 1)
    make_legend_temp(ax, ['Real', 'Predicted'], cmap=mpl.cm.viridis, fontsize=6)
    scatter_helper(e_eval_reals[mask], e_eval_preds[mask], ax=ax)
    fig.subplots_adjust(right=0.75)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    save_name_base = title.lower().replace(" ", "_")
    fig.savefig(f'{figure_path}/figure4/rna_umap_generated_{save_name_base}.png')
    fig.savefig(f'{figure_path}/figure4/rna_umap_generated_{save_name_base}.svg', format='svg')

# cluster rna space, get average clone size in each cluster (real vs predicted)
a, b = np.unique(pseudo_tcrs_holdout, return_counts=True)
srt = a.argsort()
a = a[srt]
b = b[srt]
clone_count_by_cell_pred = np.take(b, pseudo_tcrs_holdout.astype(np.int32))
clone_count_by_cell_pred = clone_count_by_cell_pred / clone_count_by_cell_pred.sum() * 1000000

clone_count_by_cell_real = np.take(df_all_tcrs.fillna(0).sum(axis=1).values, x_label[:, col_tcr].astype(np.int32))
clone_count_by_cell_real = clone_count_by_cell_real / clone_count_by_cell_real.sum() * 1000000

if do_clont_count_by_cluster:
    for k in [10, 25, 50]:
        # k = 10
        fig.set_size_inches([4.5, 3.5])
        km = sklearn.cluster.MiniBatchKMeans(k, random_state=12)
        km.fit(np.concatenate([x_rna, preds_rna_holdout], axis=0))
        clusts_real = km.predict(x_rna)
        clusts_pred = km.predict(preds_rna_holdout.astype(np.double))
        cluster_counts_real = []
        cluster_counts_pred = []
        for c in range(k):
            cluster_counts_real.append(np.median(clone_count_by_cell_real[clusts_real == c]))
            cluster_counts_pred.append(np.median(clone_count_by_cell_pred[clusts_pred == c]))

        cluster_counts_real = np.nan_to_num(np.array(cluster_counts_real), 0)
        cluster_counts_pred = np.nan_to_num(np.array(cluster_counts_pred), 0)

        r = np.corrcoef(cluster_counts_real, cluster_counts_pred)[0, 1]
        print("{:.3f}".format(r))
        model_results = sm.OLS(cluster_counts_pred, sm.add_constant(cluster_counts_real)).fit()
        fig.clf()
        ax = fig.subplots(1, 1)
        ax.scatter(cluster_counts_real, cluster_counts_pred, s=25)
        xlim = ax.get_xlim()
        ax.plot(np.arange(xlim[0], xlim[1], .01), model_results.params[0] + model_results.params[1] * np.arange(xlim[0], xlim[1], .01),
                c='k', linestyle='--', label='r = {:0.2f}'.format(np.sqrt(model_results.rsquared)))
        ax.legend(loc='lower right')
        fig.savefig(f'{figure_path}/figure4/clont_count_by_cluster_k_{k}.png')
        fig.savefig(f'{figure_path}/figure4/clont_count_by_cluster_k_{k}.svg', format='svg')
else:
    print('Skipping do_clont_count_by_cluster')


#########################
# figure 4b


def make_legend(ax, labels, s=20, cmap=mpl.cm.jet, color_order=None, center=None, **kwargs):
    if center is None:
        center = [0, 0]
    numlabs = len(np.unique(labels))
    labels_seen = set()
    for i, label in enumerate(labels):
        if label in labels_seen: continue
        if numlabs > 1:
            # ax.scatter(center[0], center[1], s=s, c=[cmap(1. * i / (numlabs - 1))], label=label)
            if color_order is not None:
                c = 1. * color_order[len(labels_seen)] / (numlabs - 1) 
            else:
                c = 1. * len(labels_seen) / (numlabs - 1)
            ax.scatter(center[0], center[1], s=s, c=[cmap(c)], label=label)
        else:
            ax.scatter(center[0], center[1], s=s, c=[cmap(1.)], label=label)
        labels_seen.add(label)
    ax.scatter(center[0], center[1], s=2 * s, c='w')
    ax.legend(**kwargs)

output_folders = [fn for fn in sorted(glob.glob('/data/che/TRIM/HNSCC/output/holdout*'))]
output_folders = [o for o in output_folders if os.path.exists(os.path.join(o, 'preds.npz'))]
all_preds = []
labels = []
for fn in output_folders:
    args = argparse.ArgumentParser().parse_args()
    with open(os.path.join(fn, 'args.txt'), 'r') as f:
        args.__dict__ = json.load(f)

    with open(os.path.join(fn, 'preds.npz'), 'rb') as f:
        npzfile = np.load(f)
        preds_rna = npzfile['preds_rna']
        preds_tcr = npzfile['preds_tcr']
        pseudo_tcrs = npzfile['pseudo_tcrs']
    
    for condition in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        mask = np.logical_and(x_label[:, col_patient] == int(args.heldout_patient),
                              np.logical_and(x_label[:, col_bloodtumor] == condition[0], x_label[:, col_prepost] == condition[1]))
        if mask.sum() == 0:
            continue
        real_diversity = x_label[mask, col_tcr]
        gen_diversity = pseudo_tcrs[mask]
        real_diversity = len(np.unique(real_diversity)) / real_diversity.shape[0]
        gen_diversity = len(np.unique(gen_diversity)) / gen_diversity.shape[0]
        all_preds.append([real_diversity, gen_diversity])
        labels.append(condition)
all_preds = np.array(all_preds)
labels = np.array(labels)

x = all_preds[:, 0]
y = all_preds[:, 1]
labels = 2 * labels[:, 0] + labels[:, 1]


model_results = sm.OLS(y, sm.add_constant(x)).fit()
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, s=15, c=labels, cmap=mpl.cm.bwr)
# Plotting the regression line
xlim = ax.get_xlim()
x_vals = np.linspace(xlim[0], xlim[1], 100)
y_vals = model_results.params[0] + model_results.params[1] * x_vals
regression_line, = ax.plot(x_vals, y_vals, c='k', linestyle='--', label=f'r = {np.sqrt(model_results.rsquared):.2f}')
# Set limits
lims = [min([ax.get_xlim()[0], ax.get_ylim()[0]]), max([ax.get_xlim()[1], ax.get_ylim()[1]])]
ax.set_xlim(lims)
ax.set_ylim(lims)
# Labels and title
ax.set_xlabel('Real diversity')
ax.set_ylabel('Generated diversity')
# Annotate beta value and r value
ax.annotate(r'$\beta_1$={:.2f}'.format(model_results.params[1]), [.77, .1], xycoords='axes fraction')
# Custom legend for regression line
first_legend = ax.legend(handles=[regression_line], loc='lower right')
# Add the first legend back to the plot
ax.add_artist(first_legend)
# Custom legend for conditions
# handles, _ = scatter.legend_elements()
# labels = ['BB', 'BA', 'TB', 'TA']
# ax.legend(handles, labels, color_order=[1, 0, 2, 3], loc='upper left')
# fig.savefig('tcr/figures/diversity_by_condition.png')

# Custom legend for conditions
handles, _ = scatter.legend_elements()
labels = ['Blood Before', 'Blood After', 'Tumor Before', 'Tumor After']
# Create the custom legend
custom_legend = [handles[i] for i in [1, 0, 2, 3]]
ax.legend(custom_legend, labels, loc='upper left')
fig.savefig(f'{figure_path}/figure4/diversity_by_condition.png')
fig.savefig(f'{figure_path}/figure4/diversity_by_condition.svg', format='svg')


#########################
# figure 4c
# expansion prediction

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

if do_fig_4c:
    baseline_expansion_prediction('rna', 'knn', 27, f'{figure_path}/figure4/expansion_prediction_baseline_knn_rna.png')
    baseline_expansion_prediction('tcr', 'knn', 27, f'{figure_path}/figure4/expansion_prediction_baseline_knn_tcr.png')
    baseline_expansion_prediction('both', 'knn', 27, f'{figure_path}/figure4/expansion_prediction_baseline_knn_both.png')
    baseline_expansion_prediction('rna', 'NN', 27, f'{figure_path}/figure4/expansion_prediction_baseline_NN_rna.png')
    baseline_expansion_prediction('tcr', 'NN', 27, f'{figure_path}/figure4/expansion_prediction_baseline_NN_tcr.png')
    baseline_expansion_prediction('both', 'NN', 27, f'{figure_path}/figure4/expansion_prediction_baseline_NN_both.png')
    # baseline_expansion_prediction('rna', 'random_forest', 27, 'figures/expansion_prediction_baseline_rf_rna.png')
    # baseline_expansion_prediction('tcr', 'random_forest', 27, 'figures/expansion_prediction_baseline_rf_tcr.png')
    # baseline_expansion_prediction('both', 'random_forest', 27, 'figures/expansion_prediction_baseline_rf_both.png')
    # baseline_expansion_prediction('rna', 'svm', 27, 'figures/expansion_prediction_baseline_svm_rna.png')
    # baseline_expansion_prediction('tcr', 'svm', 27, 'figures/expansion_prediction_baseline_svm_tcr.png')
    # baseline_expansion_prediction('both', 'svm', 27, 'figures/expansion_prediction_baseline_svm_both.png')
else:
    print('Skipping expansion prediction baseline')
    print('-'*80)

output_folders = [fn.split('/')[-1] for fn in sorted(glob.glob('/data/che/TRIM/HNSCC/output/holdout*'))]
output_folders = [o for o in output_folders if os.path.exists(os.path.join('/data/che/TRIM/HNSCC/output/', o, 'preds.npz'))]


x = all_real_clonalities[:, [0]].sum(axis=1) < all_real_clonalities[:, [1]].sum(axis=1)
y = all_pseudo_clonalities[:, [1]].sum(axis=1) / all_pseudo_clonalities[:, [0]].sum(axis=1)
y *= (all_pseudo_clonalities[:, 0] >= 2).astype(int)

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
fig.savefig(f'{figure_path}/figure4/trim_expansion_prediction.png')
fig.savefig(f'{figure_path}/figure4/trim_expansion_prediction.svg', format='svg')

# CH: List number of expanded and non-expanded clones in the test set
all_real_clonalities.shape # [number_of_unique_tcrs, 4], 4 columns are the clone counts in BB, BA, TB, TA
# 'expansion' is having a higher count after the treatment than before: BA + TA > BB + TB
number_expanded_clones = (all_real_clonalities[:, [1, 3]].sum(axis=1) > all_real_clonalities[:, [0, 2]].sum(axis=1)).sum(axis=0)
# number_expanded_clones = 1561
number_nonexpanded_clones = (all_real_clonalities[:, [1, 3]].sum(axis=1) <= all_real_clonalities[:, [0, 2]].sum(axis=1)).sum(axis=0)
# number_nonexpanded_clones = 13878
assert number_expanded_clones + number_nonexpanded_clones == all_real_clonalities.shape[0]

# BA > BB, N = 1061
(all_real_clonalities[:, 1] > all_real_clonalities[:, 0]).sum(axis=0) # 1061
# BB, N = 15439
(all_real_clonalities[:, 0] > 0).sum(axis=0) # 15439

# Some exploration for writing the paper -------------------------------------
# Number of TCRs in blood pre-treatment (get values in the first column of df_all_tcrs that are not NaN)
df_all_tcrs.iloc[:, 0].notna().sum(axis=0) # 15147
# Number of TCRs that are both in blood pre-treatment and blood post-treatment
(df_all_tcrs.iloc[:, 0].notna() & df_all_tcrs.iloc[:, 1].notna()).sum(axis=0) # 725
# Number of TCRs expanded in blood post-treatment (get values in the second column of df_all_tcrs that are not NaN and are greater than the values in the first column)
(df_all_tcrs.iloc[:, 0].notna() & df_all_tcrs.iloc[:, 1].notna() & (df_all_tcrs.iloc[:, 1] > df_all_tcrs.iloc[:, 0])).sum(axis=0) # 185 
# Number of TCRs that are both in blood pre-treatment and tumor post-treatment
(df_all_tcrs.iloc[:, 0].notna() & df_all_tcrs.iloc[:, 3].notna()).sum(axis=0) # 533
# Number of expanded TCRs in blood post-treatment (also present in blood pre-treatment) and exists in tumor post-treatment
(df_all_tcrs.iloc[:, 0].notna() & df_all_tcrs.iloc[:, 1].notna() & (df_all_tcrs.iloc[:, 1] > df_all_tcrs.iloc[:, 0]) & df_all_tcrs.iloc[:, 3].notna()).sum(axis=0) # 66
# Number of expanded TCRs in blood post-treatment (also present in blood pre-treatment) and expanded in tumor post-treatment
(df_all_tcrs.iloc[:, 0].notna() & df_all_tcrs.iloc[:, 1].notna() & (df_all_tcrs.iloc[:, 1] > df_all_tcrs.iloc[:, 0]) & (df_all_tcrs.iloc[:, 3].fillna(0) > df_all_tcrs.iloc[:, 2].fillna(0))).sum(axis=0) # 64
# Number of expanded TCRs in blood post-treatment (also present in blood pre-treatment) and expanded in tumor post-treatment (emergent)
(df_all_tcrs.iloc[:, 0].notna() & df_all_tcrs.iloc[:, 1].notna() & (df_all_tcrs.iloc[:, 1] > df_all_tcrs.iloc[:, 0]) & (df_all_tcrs.iloc[:, 3].fillna(0) > df_all_tcrs.iloc[:, 2].fillna(0)) & ~df_all_tcrs.iloc[:, 2].notna()).sum(axis=0) # 62
# -----------------------------------------------------------------------------


# how accurate are clonality counts?
x = all_real_clonalities[:, [0, 1]].sum(axis=1)
y = all_pseudo_clonalities[:, [0, 1]].sum(axis=1)

x = np.log(x + 1e-6)
y = np.log(y + 1e-6)

model_results = sm.OLS(y, sm.add_constant(x.astype(int))).fit()
print(np.sqrt(model_results.rsquared))
fig.clf()
ax = fig.subplots(1, 1)
ax.scatter(x, y, s=5, c='b', alpha=.5)
ax.plot(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], .01), model_results.params[0] + model_results.params[1] * np.arange(ax.get_xlim()[0], ax.get_xlim()[1], .01), c='k', linestyle='--')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Real Clonality (Log) Count')
ax.set_ylabel('Psuedoclonality (Log) Count')
# fig.savefig(f'{figure_path}/figure4/clone_count_tmp.png')

#########################


#########################
# figure 4d

np.float = float # Patch for compatibility with older libraries
def get_differentially_expressed_genes(df, labels, expressed_ratio=.1):
    import scanpy
    import diffxpy.api as de
    adata = scanpy.AnnData(df, df.index.to_frame(), df.columns.to_frame())
    adata.obs['groups'] = labels
    out = de.test.t_test(adata, grouping='groups')

    unique_labels = np.unique(labels)
    cells_expressed_in = []
    for unique_label in unique_labels:
        ratio = (df[labels == unique_label] > 0).sum(axis=0) / df[labels == unique_label].shape[0]
        cells_expressed_in.append(ratio > expressed_ratio)
    cells_expressed_in = pd.DataFrame(cells_expressed_in)
    cells_expressed_in = cells_expressed_in.all(axis=0)

    genes_sorted = out.log_fold_change().argsort()
    genes_sorted = [g for g in genes_sorted if cells_expressed_in[g]]

    return genes_sorted

def get_diffexp_genes_by_expanded(cell_tcrs, tcr_counts, tmp_x_label, tmp_preds_rna_ambient, delta=0):
    rna_bb_expanded = []
    rna_bb_nonexpanded = []
    mask_expanded = np.logical_and((tcr_counts[:, 0] / tcr_counts[:, 1]) < (1 - delta), tcr_counts[:, :2].sum(axis=-1) > 0)
    mask_nonexpanded = np.logical_and((tcr_counts[:, 0] / tcr_counts[:, 1]) > (1 + delta), tcr_counts[:, :2].sum(axis=-1) > 0)
    for tcr_id in range(tcr_counts.shape[0]):
        if mask_expanded[tcr_id]:
            rna_mask = np.logical_and(tmp_x_label[:, col_bloodtumor] == 0, tmp_x_label[:, col_prepost] == 0)
            rna_mask = np.logical_and(rna_mask, cell_tcrs == tcr_id)
            rna_bb_expanded.append(tmp_preds_rna_ambient[rna_mask])
        elif mask_nonexpanded[tcr_id]:
            rna_mask = np.logical_and(tmp_x_label[:, col_bloodtumor] == 0, tmp_x_label[:, col_prepost] == 0)
            rna_mask = np.logical_and(rna_mask, cell_tcrs == tcr_id)
            rna_bb_nonexpanded.append(tmp_preds_rna_ambient[rna_mask])

    rna_bb_expanded = np.concatenate(rna_bb_expanded, axis=0)
    rna_bb_nonexpanded = np.concatenate(rna_bb_nonexpanded, axis=0)

    diffexp_genes = get_differentially_expressed_genes(
                        pd.DataFrame(np.concatenate([rna_bb_expanded, rna_bb_nonexpanded], axis=0)),
                        np.concatenate([np.zeros(rna_bb_expanded.shape[0]), np.ones(rna_bb_nonexpanded.shape[0])], axis=0).astype(str),
                        expressed_ratio=.5)
    return diffexp_genes


pseudo_tcrs_counts = np.zeros([int(pseudo_tcrs_holdout.max() + 1), 4])
for ii, [f1, f2] in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
    a, b = np.unique(pseudo_tcrs_holdout[np.logical_and(x_label[:, col_bloodtumor] == f1, x_label[:, col_prepost] == f2)], return_counts=True)
    for ai, bi in zip(a, b):
        pseudo_tcrs_counts[int(ai), ii] = bi

diffexp_genes_pseudo = get_diffexp_genes_by_expanded(pseudo_tcrs_holdout, pseudo_tcrs_counts, x_label, preds_rna_ambient)
diffexp_genes_real = get_diffexp_genes_by_expanded(x_label[:, col_tcr], df_all_tcrs.fillna(0).values, x_label, data_rna_ambient)
diffexp_genes_pseudo = [g for g in diffexp_genes_pseudo if g in diffexp_genes_real]
diffexp_genes_real = [g for g in diffexp_genes_real if g in diffexp_genes_pseudo]


ks = np.array(range(5, len(diffexp_genes_real), 5))
proportions = []
for k in ks:
    u = set(diffexp_genes_pseudo[:k]).intersection(set(diffexp_genes_real[:k]))
    proportions.append(1. * len(u) / k)
proportions = np.array(proportions)
auc = proportions.sum() / proportions.shape[0]

# def get_null_baseline():
    # null_baseline_proportions = []
    # for _ in range(25):
    #     np.random.shuffle(diffexp_genes_real)
    #     np.random.shuffle(diffexp_genes_pseudo)
    #     ks = np.array(range(5, len(diffexp_genes_real), 5))
    #     proportions = []
    #     for k in ks:
    #         u = set(diffexp_genes_pseudo[:k]).intersection(set(diffexp_genes_real[:k]))
    #         proportions.append(1. * len(u) / k)
    #     proportions = np.array(proportions)
    #     null_baseline_proportions.append(proportions)
    #     auc = proportions.sum() / proportions.shape[0]
    #     print(_, auc)
    # null_baseline_proportions_mean = np.stack(null_baseline_proportions, axis=0).mean(0)

fig.clf()
ax = fig.subplots(1, 1)
# ax.scatter(ks, proportions, c='k', s=1)
ax.plot(ks, proportions, color="darkorange", lw=2, label="AUC = %0.2f" % auc)

xlim, ylim = [[0, ks.max()], [0, 1]]
ax.plot(xlim, ylim, color="navy", lw=2, linestyle="--")
[ax.set_xlim(xlim), ax.set_ylim(ylim)]
ax.legend(loc='lower right')
ax.set_xlabel('k')
ax.set_ylabel('Proportion of intersection')
ax.set_xticks(range(0, ks.max(), 2500))
fig.savefig(f'{figure_path}/figure5/gene_signature_diffexp.png')
fig.savefig(f'{figure_path}/figure5/gene_signature_diffexp.svg', format='svg')


########################
## repeat for subsets:
# cell_type = 'cd8'
for cell_type in ['cd4', 'cd8']:
    if cell_type == 'cd4':
        mask = x_label[:, col_celltype] == 0
    elif cell_type == 'cd8':
        mask = x_label[:, col_celltype] == 1
    diffexp_genes_pseudo = get_diffexp_genes_by_expanded(pseudo_tcrs_holdout[mask], pseudo_tcrs_counts, x_label[mask], preds_rna_ambient[mask])
    diffexp_genes_real = get_diffexp_genes_by_expanded(x_label[:, col_tcr][mask], df_all_tcrs.fillna(0).values, x_label[mask], data_rna_ambient[mask])
    diffexp_genes_pseudo = [g for g in diffexp_genes_pseudo if g in diffexp_genes_real]
    diffexp_genes_real = [g for g in diffexp_genes_real if g in diffexp_genes_pseudo]

    ks = np.array(range(5, len(diffexp_genes_real), 5))
    proportions = []
    for k in ks:
        u = set(diffexp_genes_pseudo[:k]).intersection(set(diffexp_genes_real[:k]))
        proportions.append(1. * len(u) / k)
    proportions = np.array(proportions)
    auc = proportions.sum() / proportions.shape[0]

    fig.clf()
    ax = fig.subplots(1, 1)
    ax.plot(ks, proportions, color="darkorange", lw=2, label="AUC = %0.2f" % auc)
    xlim, ylim = [[0, ks.max()], [0, 1]]
    ax.plot(xlim, ylim, color="navy", lw=2, linestyle="--")
    [ax.set_xlim(xlim), ax.set_ylim(ylim)]
    ax.legend(loc='lower right')
    ax.set_xlabel('k')
    ax.set_ylabel('Proportion of intersection')
    ax.set_xticks(range(0, ks.max(), 2500))
    fig.savefig(f'{figure_path}/figure5/gene_signature_diffexp_subset_{cell_type}.png')
    fig.savefig(f'{figure_path}/figure5/gene_signature_diffexp_subset_{cell_type}.svg', format='svg')

########################

########################
## heatmap of top 100 genes


def get_diffexp_genes_by_expanded_prepost(cell_tcrs, tcr_counts, tmp_x_label, tmp_preds_rna_ambient, delta=0):
    rna_bb_expanded = []
    rna_ba_expanded = []
    mask_expanded = np.logical_and((tcr_counts[:, 0] / tcr_counts[:, 1]) < (1 - delta), tcr_counts[:, :2].sum(axis=-1) > 0)
    # mask_nonexpanded = np.logical_and((tcr_counts[:, 0] / tcr_counts[:, 1]) > (1 + delta), tcr_counts[:, :2].sum(axis=-1) > 0)
    for tcr_id in range(tcr_counts.shape[0]):
        if mask_expanded[tcr_id]:
            rna_mask = np.logical_and(tmp_x_label[:, col_bloodtumor] == 0, tmp_x_label[:, col_prepost] == 0)
            rna_mask = np.logical_and(rna_mask, cell_tcrs == tcr_id)
            rna_bb_expanded.append(tmp_preds_rna_ambient[rna_mask])
        # elif mask_nonexpanded[tcr_id]:
            rna_mask = np.logical_and(tmp_x_label[:, col_bloodtumor] == 0, tmp_x_label[:, col_prepost] == 1)
            rna_mask = np.logical_and(rna_mask, cell_tcrs == tcr_id)
            rna_ba_expanded.append(tmp_preds_rna_ambient[rna_mask])

    rna_bb_expanded = np.concatenate(rna_bb_expanded, axis=0)
    rna_ba_expanded = np.concatenate(rna_ba_expanded, axis=0)

    diffexp_genes = get_differentially_expressed_genes(
                        pd.DataFrame(np.concatenate([rna_bb_expanded, rna_ba_expanded], axis=0)),
                        np.concatenate([np.zeros(rna_bb_expanded.shape[0]), np.ones(rna_ba_expanded.shape[0])], axis=0).astype(str),
                        expressed_ratio=.2)
    return diffexp_genes, rna_bb_expanded, rna_ba_expanded

diffexp_genes_pseudo, rna_bb, rna_ba = get_diffexp_genes_by_expanded_prepost(pseudo_tcrs_holdout, pseudo_tcrs_counts, x_label, preds_rna_ambient)


heatmap_genes = diffexp_genes_pseudo[-50:]
tmpdata = np.concatenate([rna_bb, rna_ba], axis=0)[:, heatmap_genes].T

rna_bb_tmp = (rna_bb[:, heatmap_genes].T - tmpdata.mean(axis=1, keepdims=True)) / tmpdata.std(axis=1, keepdims=True)
rna_ba_tmp = (rna_ba[:, heatmap_genes].T - tmpdata.mean(axis=1, keepdims=True)) / tmpdata.std(axis=1, keepdims=True)

# top 50 DE genes
gene_list = [combined_data_columns[col] for col in heatmap_genes]
# save the gene list to a file
with open(f'{figure_path}/figure5/de_50_genes.txt', 'w') as f:
    for gene in gene_list:
        f.write(f"{gene}\n")

# timepoint = 'blood_before'
for timepoint in ['blood_before', 'blood_after']:
    if timepoint == 'blood_before':
        g = sns.clustermap(rna_bb_tmp, col_cluster=True, row_cluster=False, vmin=np.percentile(rna_bb_tmp, 1), vmax=np.percentile(rna_bb_tmp, 99))
    elif timepoint == 'blood_after':
        g = sns.clustermap(rna_ba_tmp, col_cluster=True, row_cluster=False, vmin=np.percentile(rna_ba_tmp, 1), vmax=np.percentile(rna_ba_tmp, 99))
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_yticks([t + .5 for t in range(len(heatmap_genes))])
    g.ax_heatmap.set_yticklabels([combined_data_columns[col] for col in heatmap_genes], rotation=0)
    g.cax.set_visible(False)
    g.fig.subplots_adjust(.05, .05, .85, .85)
    g.fig.savefig(f'{figure_path}/figure5/de_50_heatmap_{timepoint}.png')
    g.fig.savefig(f'{figure_path}/figure5/de_50_heatmap_{timepoint}.svg', format='svg')


# Compute mean expression across cells (genes as rows)
mean_expr_before = rna_bb_tmp.mean(axis=1)
mean_expr_after = rna_ba_tmp.mean(axis=1)

# Combine into a DataFrame: genes x [before, after]
mean_expr_mat = pd.DataFrame({
    'blood_before': mean_expr_before,
    'blood_after': mean_expr_after
})

# Plot heatmap
plt.figure(figsize=(4, len(mean_expr_mat) * 0.2))  # Dynamic height
sns.heatmap(mean_expr_mat, cmap='vlag', yticklabels=True, xticklabels=True, cbar_kws={"label": "z-score"})
plt.title("Gene expression (mean per gene, z-scored)")
plt.tight_layout()
plt.savefig(f"{figure_path}/figure5/de_50_gene_mean_expression_heatmap.png", dpi=300)
plt.savefig(f"{figure_path}/figure5/de_50_gene_mean_expression_heatmap.svg")
plt.show()

########################

# These lists were gathered from Fig.4E and SupFig.4G
signature_genes_exp = ['KLRD1', 'CXCL13', 'GNLY', 'KLRC2', 'CD7', 'ITGAE', 'GZMB', 'CTLA4', 'ZNF683', 'TNFRSF18', 'CD226', 'SIRPG', 'KLRB1', 'PRF1', 'TIGIT', 'LAG3', 'GZMA', 'IFNG']
signature_genes_nonexp = ['GZMK', 'CST7', 'EOMES', 'GZMM', 'SAMD3', 'LIME1', 'CXCR3', 'SH2D1A', 'TRAT1', 'CD5', 'IL7R', 'LEPROTL1', 'PDCD4']
# signature_genes_exp = ['GZMB', 'ALOX5AP', 'ENTPD1', 'CTLA4', 'LINC01871', 'CXCL13', 'ITGAE', 'RBPJ', 'GPR25', 'CD7', 'PTMS', 'PHLDA1', 'ACP5', 'CSF1', 'CAPG', 'GNLY', 'LINC02446', 'CCL3', 'HOPX']
# signature_genes_nonexp = ['GZMK', 'CXCR4', 'DUSP2', 'IL7R', 'KLF2', 'LYAR', 'TXNIP', 'CST7', 'MT-ND1', 'TC2N', 'GPR183', 'JUNB', 'ZFP36', 'LTB', 'CLDND1', 'PIK3R1', 'ANXA1', 'FOS', 'FOSB', 'CMC1']
# signature_genes_exp = ['GZMB', 'CTLA4', 'CXCL13', 'CD7', 'ITGAE', 'GZMA', 'GNLY', 'SIRPG', 'KLRD1', 'IFNG', 'ZNF683']
# signature_genes_nonexp = ['GZMK', 'IL7R', 'CST7', 'LEPROTL1', 'SAMD3', 'PDCD4']

inds_exp = []
genes_exp = []
inds_nonexp = []
genes_nonexp = []
for gene in signature_genes_exp:
    if gene in combined_data_columns:
        i = combined_data_columns.tolist().index(gene)
        if i in diffexp_genes_pseudo:
            print('{:<8}  {:>5}'.format(gene, diffexp_genes_pseudo.index(i)))
            inds_exp.append(diffexp_genes_pseudo.index(i))
            genes_exp.append(gene)
        else:
            print('{:<8}     -'.format(gene))
    else:
        print('{:<8}     -'.format(gene))
print('\n'*2)
for gene in signature_genes_nonexp:
    if gene in combined_data_columns:
        i = combined_data_columns.tolist().index(gene)
        if i in diffexp_genes_pseudo:
            print('{:<8}  {:>5}'.format(gene, diffexp_genes_pseudo.index(i)))
            inds_nonexp.append(diffexp_genes_pseudo.index(i))
            genes_nonexp.append(gene)
        else:
            print('{:<8}     -'.format(gene))
    else:
        print('{:<8}     -'.format(gene))

p_value = scipy.stats.ranksums(inds_exp, inds_nonexp, alternative='less').pvalue
print(np.median(inds_exp))
print(np.median(inds_nonexp))
print()


y = np.array(inds_exp + inds_nonexp)
c = np.array([mpl.cm.viridis(1.)] * len(inds_exp) + [mpl.cm.viridis(0.)] * len(inds_nonexp))
r = np.argsort(y)
for i, gene in enumerate(y[r]):
    gene_name = combined_data_columns[diffexp_genes_pseudo[gene]]
    color = 'blue' if gene_name in signature_genes_exp else 'red'
    print(termcolor.colored("{}. {}".format(i + 1, gene_name), color))
print('Ranksum p-value = {:.3f}'.format(p_value))
# print("AUC: {:.3f}".format(auc))



#########################
# figure 4e
col_hladr = 15740
col_cd38 = 11098

def get_hladr_cd38(data_pca, data_ambient, prepost):
    thresh_hladr = np.percentile(data_ambient[:, col_hladr], 70)
    thresh_cd38 = np.percentile(data_ambient[:, col_cd38], 70)
    pcts = []
    for i_pid in range(27):
        mask = x_label[:, col_patient] == i_pid
        mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 0)
        mask = np.logical_and(mask, x_label[:, col_prepost] == prepost)

        if mask.sum() == 0:
            pcts.append(np.nan)
            continue

        mask_cd8 = classifier_cd8.predict(data_pca[mask]) == 1

        tmp = data_ambient[mask][mask_cd8]
        if tmp.shape[0] <= 10:
            pcts.append(np.nan)
            continue

        mask2 = np.logical_and(tmp[:, col_hladr] > thresh_hladr, tmp[:, col_cd38] > thresh_cd38)
        pct = mask2.sum() / tmp.shape[0]
        pcts.append(pct)
        print("{:>2} {:>3}".format(i_pid, tmp.shape[0]),  "{:.2f}".format(pct))

    p_matrix = pd.crosstab(df_blood_metadata['path_downstage'], df_blood_metadata['patient']).T.loc[patient_ids].values
    pcts[np.argwhere(p_matrix.argmax(-1) == 0)[0, 0]] = np.nan
    c = p_matrix[:, 1:].argmax(-1)

    tmp = np.array([[p, label] for label, p in zip(c, pcts)])#if p is not None])

    return tmp

tmp_preds_pre = get_hladr_cd38(preds_rna_holdout, preds_rna_ambient, 0)
tmp_preds_post = get_hladr_cd38(preds_rna_holdout, preds_rna_ambient, 1)

tmp_real_pre = get_hladr_cd38(x_rna, data_rna_ambient, 0)
tmp_real_post = get_hladr_cd38(x_rna, data_rna_ambient, 1)

mask = [all([not np.isnan(p) for p in p_list]) for p_list in zip(tmp_preds_post[:, 0], tmp_preds_pre[:, 0], tmp_real_post[:, 0], tmp_real_pre[:, 0])]

tmp_pred = np.concatenate([tmp_preds_pre[mask, 0], tmp_preds_post[mask, 0]], axis=0)
tmp_real = np.concatenate([tmp_real_pre[mask, 0], tmp_real_post[mask, 0]], axis=0)

print(np.corrcoef(tmp_real_pre[mask, 0], tmp_real_post[mask, 0])[0, 1])
print(np.corrcoef(tmp_preds_pre[mask, 0], tmp_preds_post[mask, 0])[0, 1])
print(np.corrcoef(tmp_pred, tmp_real)[0, 1])


# scale = max([tmp_preds_post[mask, 0].max(), tmp_preds_pre[mask, 0].max()]) / max([tmp_real_post[mask, 0].max(), tmp_real_pre[mask, 0].max()])
# tmp_real_pre[:, 0] *= scale 
# tmp_real_post[:, 0] *= scale 

fig.set_size_inches([3.5, 6])
fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(sum(mask) * [0], tmp_preds_pre[mask, 0], c='k', s=15)
ax1.scatter(sum(mask) * [1], tmp_preds_post[mask, 0], c='k', s=15)
# ax1.scatter(sum(mask) * [-.1], tmp_real_pre[mask, 0], c='k', s=15)
# ax1.scatter(sum(mask) * [1.1], tmp_real_post[mask, 0], c='k', s=15)

# tmp_plotx = np.stack([np.zeros(sum(mask)), np.ones(sum(mask))], axis=-1)
# tmp_ploty = np.stack([tmp_preds_pre[mask, 0], tmp_preds_post[mask, 0]], axis=-1)
# c = np.where( tmp_ploty[:, 0] < tmp_ploty[:, 1], 'g', 'r').tolist()
# for i in range(tmp_plotx.shape[0]):
#     ax1.plot(tmp_plotx[i].T, tmp_ploty[i].T, c=c[i], linestyle='--')
# tmp_plotx = np.stack([-.1 * np.ones(sum(mask)), 0 * np.ones(sum(mask))], axis=-1)
# tmp_ploty = np.stack([tmp_real_pre[mask, 0], tmp_preds_pre[mask, 0]], axis=-1)
# for i in range(tmp_plotx.shape[0]):
#     ax1.plot(tmp_plotx[i].T, tmp_ploty[i].T, c='k', linestyle='-')
# tmp_plotx = np.stack([1 * np.ones(sum(mask)), 1.1 * np.ones(sum(mask))], axis=-1)
# tmp_ploty = np.stack([tmp_preds_post[mask, 0], tmp_real_post[mask, 0]], axis=-1)
# for i in range(tmp_plotx.shape[0]):
#     ax1.plot(tmp_plotx[i].T, tmp_ploty[i].T, c='k', linestyle='-')

ax1.plot([-.1, .1], 2 * [np.mean(tmp_preds_pre[mask, 0])], c='k', linestyle='--')
ax1.plot([.9, 1.1], 2 * [np.mean(tmp_preds_post[mask, 0])], c='k', linestyle='--')

for ax in [ax1]:
    ax.set_xlim([-.25, 1.25])
    ax.set_xticks([0, 1])
    # ax.set_xticklabels(['True', 'Predicted', 'Predicted','True'], rotation=30, fontsize=8)
    ax.set_xticklabels(['Pre-treatment', 'Post-treatment'])
    # ax.set_ylabel('Pct of all T cells that are CD38+/HLA-DR+')
fig.savefig(f'{figure_path}/figure4/prepost_vs_blood_activated.png')
fig.savefig(f'{figure_path}/figure4/prepost_vs_blood_activated.svg', format='svg')
fig.set_size_inches(old_fig_size)


fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(tmp_pred, tmp_real, c='k', s=15)
lims = [min([ax1.get_xlim()[0], ax1.get_ylim()[0]]), max([ax1.get_xlim()[1], ax1.get_ylim()[1]])]
ax1.set_xlim(lims)
ax1.set_ylim(lims)

model_results = sm.OLS(tmp_real, sm.add_constant(tmp_pred)).fit()
a = model_results.params[1]
b = model_results.params[0]
ax1.plot(np.arange(lims[0], lims[1], .01), a * np.arange(lims[0], lims[1], .01) + b, c='k', linestyle='--', label='r = {:0.2f}'.format(np.sqrt(model_results.rsquared)))

ax1.legend(loc='lower right')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Real')
ax1.annotate(r'$\beta_1$={:.2f}'.format(model_results.params[1]), [.77, .1], xycoords='axes fraction')
fig.savefig(f'{figure_path}/figure4/prepost_vs_blood_activated_dotplot.png')
fig.savefig(f'{figure_path}/figure4/prepost_vs_blood_activated_dotplot.svg', format='svg')


#########################
# figure 4d
col_pd1 = 21953  # PD-1 / PDCD1
col_klrg1 = 17273

def get_klrg1(data_pca, data_ambient):
    thresh_pd1 = np.percentile(data_ambient[:, col_pd1], 30)
    # thresh_klrg1 = np.percentile(data_ambient[:, col_klrg1], 50)
    pcts = []
    for i_pid in range(27):
        mask = x_label[:, col_patient] == i_pid
        mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 0)
        # mask = np.logical_and(mask, x_label[:, col_prepost] == 1)
        mask = np.logical_and(mask, data_ambient[:, col_pd1] > thresh_pd1)

        if mask.sum() == 0:
            pcts.append(np.nan)
            continue

        mask_cd8 = classifier_cd8.predict(data_pca[mask]) == 1
        # mask_cd8 = x_label[mask, col_celltype] == 1

        tmp = data_ambient[mask][mask_cd8]
        if tmp.shape[0] <= 10:
            pcts.append(np.nan)
            continue

        # pct = (tmp[:, col_klrg1] < thresh_klrg1).sum() / tmp.shape[0]
        pct = tmp[:, col_klrg1].mean()
        pcts.append(pct)
        print("{:>2} {:>3}".format(i_pid, tmp.shape[0]),  "{:.2f}".format(pct))

    p_matrix = pd.crosstab(df_blood_metadata['path_downstage'], df_blood_metadata['patient']).T.loc[patient_ids].values
    c = p_matrix[:, 1:].argmax(-1)

    tmp = np.array([[p, label] for label, p in zip(c, pcts)])# if p is not None])

    return tmp


tmp_real = get_klrg1(x_rna, data_rna_ambient)
tmp_pred = get_klrg1(preds_rna_holdout, preds_rna_ambient)

mask = [all([not np.isnan(p) for p in p_list]) for p_list in zip(tmp_real[:, 0], tmp_pred[:, 0])]

print(np.corrcoef(tmp_real[mask, 0], tmp_real[mask, 1])[0, 1])
print(np.corrcoef(tmp_pred[mask, 0], tmp_pred[mask, 1])[0, 1])
print(np.corrcoef(tmp_real[mask, 0], tmp_pred[mask, 0])[0, 1])

fig.set_size_inches(old_fig_size)
fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(tmp_pred[mask, 0], tmp_real[mask, 0], c='k', s=15)

lims = [min([ax1.get_xlim()[0], ax1.get_ylim()[0]]), max([ax1.get_xlim()[1], ax1.get_ylim()[1]])]
ax1.set_xlim(lims)
ax1.set_ylim(lims)

model_results = sm.OLS(tmp_real[mask, 0], sm.add_constant(tmp_pred[mask, 0])).fit()
a = model_results.params[1]
b = model_results.params[0]
ax1.plot(np.arange(lims[0], lims[1], .01), a * np.arange(lims[0], lims[1], .01) + b, c='k', linestyle='--', label='r = {:0.2f}'.format(np.sqrt(model_results.rsquared)))

ax1.legend(loc='lower right')
# ax1.set_xlabel('Predicted')
# ax1.set_ylabel('Real')
ax1.annotate(r'$\beta_1$={:.2f}'.format(model_results.params[1]), [.77, .1], xycoords='axes fraction')
fig.savefig(f'{figure_path}/figure4/response_vs_blood_pd1_klrg1_dotplot.png')
fig.savefig(f'{figure_path}/figure4/response_vs_blood_pd1_klrg1_dotplot.svg', format='svg')

fig.set_size_inches([3.5, 6])
fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(sum(tmp_pred[mask, 1] == 0) * [0], tmp_pred[mask, 0][tmp_pred[mask, 1] == 0], c='k', s=15)
ax1.scatter(sum(tmp_pred[mask, 1] == 1) * [1], tmp_pred[mask, 0][tmp_pred[mask, 1] == 1], c='k', s=15)

ax1.plot([-.1, .1], 2 * [np.mean(tmp_pred[mask, 0][tmp_pred[mask, 1] == 0])], c='k', linestyle='--')
ax1.plot([.9, 1.1], 2 * [np.mean(tmp_pred[mask, 0][tmp_pred[mask, 1] == 1])], c='k', linestyle='--')

ax1.plot()
ax1.set_xlim([-.25, 1.25])
ax1.set_xticks([0, 1])
ax1.set_xlabel('Pathological response')
ax1.set_xticklabels(['No', 'Yes'])
ax1.set_ylabel('KLRG1 mean expression')
# set y labels to be 2 decimal places
ax1.set_yticklabels(['{:.2f}'.format(y) for y in ax1.get_yticks()])

fig.savefig(f'{figure_path}/figure4/response_vs_blood_pd1_klrg1.png')
fig.savefig(f'{figure_path}/figure4/response_vs_blood_pd1_klrg1.svg', format='svg')
fig.set_size_inches(old_fig_size)


#########################
# figure 4e
col_hladr = 15740
col_cd38 = 11098


def get_hladr_cd38(data_pca, data_ambient, prepost):
    thresh_hladr = np.percentile(data_ambient[:, col_hladr], 60)
    thresh_cd38 = np.percentile(data_ambient[:, col_cd38], 75)
    pcts = []
    for i_pid in range(27):
        mask = x_label[:, col_patient] == i_pid
        mask = np.logical_and(mask, x_label[:, col_bloodtumor] == 0)
        mask = np.logical_and(mask, x_label[:, col_prepost] == prepost)

        if mask.sum() == 0:
            pcts.append(np.nan)
            continue

        mask_cd8 = classifier_cd8.predict(data_pca[mask]) == 1
        # mask_cd8 = x_label[mask, col_celltype] == 1

        tmp = data_ambient[mask][mask_cd8]
        if tmp.shape[0] <= 10:
            pcts.append(np.nan)
            continue

        mask2 = np.logical_and(tmp[:, col_hladr] > thresh_hladr, tmp[:, col_cd38] > thresh_cd38)
        pct = mask2.sum() / tmp.shape[0]
        pcts.append(pct)
        print("{:>2} {:>3}".format(i_pid, tmp.shape[0]),  "{:.2f}".format(pct))

    p_matrix = pd.crosstab(df_blood_metadata['path_downstage'], df_blood_metadata['patient']).T.loc[patient_ids].values
    pcts[np.argwhere(p_matrix.argmax(-1) == 0)[0, 0]] = np.nan
    c = p_matrix[:, 1:].argmax(-1)

    tmp = np.array([[p, label] for label, p in zip(c, pcts)])# if p is not None])

    return tmp

tmp_real_pre = get_hladr_cd38(x_rna, data_rna_ambient, 0)
tmp_real_post = get_hladr_cd38(x_rna, data_rna_ambient, 1)

tmp_preds_pre = get_hladr_cd38(preds_rna_holdout, preds_rna_ambient, 0)
tmp_preds_post = get_hladr_cd38(preds_rna_holdout, preds_rna_ambient, 1)

mask = [all([not np.isnan(p) for p in p_list]) for p_list in zip(tmp_preds_post[:, 0], tmp_preds_pre[:, 0], tmp_real_post[:, 0], tmp_real_pre[:, 0])]
# scale = max([tmp_preds_pre[mask, 0].max(), tmp_preds_pre[mask, 0].max()]) / max([tmp_real_pre[mask, 0].max(), tmp_real_pre[mask, 0].max()])
# tmp_real_pre[:, 0] *= scale 
# tmp_real_post[:, 0] *= scale 

print(np.corrcoef(tmp_real_pre[mask, 0], tmp_real_pre[mask, 1])[0, 1])
print(np.corrcoef(tmp_real_post[mask, 0], tmp_real_post[mask, 1])[0, 1])
print(np.corrcoef(tmp_preds_pre[mask, 0], tmp_preds_pre[mask, 1])[0, 1])
print(np.corrcoef(tmp_preds_post[mask, 0], tmp_preds_post[mask, 1])[0, 1])


# fig.clf()
# ax1 = fig.subplots(1, 1)
# ax1.scatter(sum(tmp_preds_pre[mask, 1] == 0) * [0], tmp_preds_pre[mask, 0][tmp_preds_pre[mask, 1] == 0], c='k', s=15)
# ax1.scatter(sum(tmp_preds_pre[mask, 1] == 1) * [1], tmp_preds_pre[mask, 0][tmp_preds_pre[mask, 1] == 1], c='k', s=15)
# ax1.scatter(sum(tmp_real_pre[mask, 1] == 0) * [-.1], tmp_real_pre[mask, 0][tmp_real_pre[mask, 1] == 0], c='k', s=15)
# ax1.scatter(sum(tmp_real_pre[mask, 1] == 1) * [1.1], tmp_real_pre[mask, 0][tmp_real_pre[mask, 1] == 1], c='k', s=15)


# tmp_plotx = np.stack([-.1 * np.ones(sum(tmp_preds_pre[mask, 1] == 0)), 0 * np.ones(sum(tmp_preds_pre[mask, 1] == 0))], axis=-1)
# tmp_ploty = np.stack([tmp_real_pre[mask, 0][tmp_real_pre[mask, 1] == 0], tmp_preds_pre[mask, 0][tmp_preds_pre[mask, 1] == 0]], axis=-1)
# for i in range(tmp_plotx.shape[0]):
#     ax1.plot(tmp_plotx[i].T, tmp_ploty[i].T, c='k', linestyle='--', linewidth=1)

# tmp_plotx = np.stack([1 * np.ones(sum(tmp_preds_pre[mask, 1] == 1)), 1.1 * np.ones(sum(tmp_preds_pre[mask, 1] == 1))], axis=-1)
# tmp_ploty = np.stack([tmp_preds_pre[mask, 0][tmp_preds_pre[mask, 1] == 1], tmp_real_pre[mask, 0][tmp_real_pre[mask, 1] == 1]], axis=-1)
# for i in range(tmp_plotx.shape[0]):
#     ax1.plot(tmp_plotx[i].T, tmp_ploty[i].T, c='k', linestyle='--', linewidth=1)


# for ax in [ax1]:
#     ax.set_xlim([-.25, 1.25])
#     ax.set_xticks([-.1, 0, 1,1.1])
#     ax.set_xticklabels(['True', 'Predicted', 'Predicted','True'], rotation=30, fontsize=8)
#     # ax.set_xlabel('Pathological response')
#     # ax.set_xticklabels(['No', 'Yes'])
#     ax.set_ylabel('Pct of CD8+ T cells that are CD38+/HLA-DR+')
# fig.savefig('figures/response_vs_blood_activated.png')


tmp_pred = np.concatenate([tmp_preds_pre[mask, 0], tmp_preds_post[mask, 0]], axis=0)
tmp_real = np.concatenate([tmp_real_pre[mask, 0], tmp_real_post[mask, 0]], axis=0)

print(np.corrcoef(tmp_pred, tmp_real)[0, 1])

fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(tmp_pred, tmp_real, c='k', s=15)


lims = [min([ax1.get_xlim()[0], ax1.get_ylim()[0]]), max([ax1.get_xlim()[1], ax1.get_ylim()[1]])]
ax1.set_xlim(lims)
ax1.set_ylim(lims)

model_results = sm.OLS(tmp_real, sm.add_constant(tmp_pred)).fit()
a = model_results.params[1]
b = model_results.params[0]
ax1.plot(np.arange(lims[0], lims[1], .01), a * np.arange(lims[0], lims[1], .01) + b, c='k', linestyle='--', label='r = {:0.2f}'.format(np.sqrt(model_results.rsquared)))

ax1.legend(loc='lower right')
# ax1.set_xlabel('Predicted')
# ax1.set_ylabel('Real')
ax1.annotate(r'$\beta_1$={:.2f}'.format(model_results.params[1]), [.77, .1], xycoords='axes fraction')
fig.savefig(f'{figure_path}/figure4/response_vs_blood_activated_dotplot.png')
fig.savefig(f'{figure_path}/figure4/response_vs_blood_activated_dotplot.svg', format='svg')


fig.set_size_inches([3.5, 6])
fig.clf()
ax1 = fig.subplots(1, 1)
ax1.scatter(sum(tmp_preds_pre[mask, 1] == 0) * [0], tmp_preds_pre[mask, 0][tmp_preds_pre[mask, 1] == 0], c='k', s=15)
ax1.scatter(sum(tmp_preds_pre[mask, 1] == 1) * [1], tmp_preds_pre[mask, 0][tmp_preds_pre[mask, 1] == 1], c='k', s=15)

mean_no_pre_pred = np.median(tmp_preds_pre[mask][tmp_preds_pre[mask, 1] == 0, 0])
mean_yes_pre_pred = np.median(tmp_preds_pre[mask][tmp_preds_pre[mask, 1] == 1, 0])
ax1.plot([-.1, .1], 2 * [mean_no_pre_pred], c='k', linestyle='--')
ax1.plot([.9, 1.1], 2 * [mean_yes_pre_pred], c='k', linestyle='--')
for ax in [ax1]:
    ax.set_xlim([-.25, 1.25])
    ax.set_xticks([0, 1])
    # ax.set_xticklabels(['True', 'Predicted', 'Predicted','True'], rotation=30, fontsize=8)
    ax.set_xlabel('Pathological response')
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylabel('Pct of CD8+ T cells that are CD38+/HLA-DR+')
fig.savefig(f'{figure_path}/figure4/response_vs_blood_activated.png')
fig.savefig(f'{figure_path}/figure4/response_vs_blood_activated.svg', format='svg')
fig.set_size_inches(old_fig_size)



#############################################
#############################################
# not yet in any figure


#########################
# to plot cd4 naive / cd8 naive to see what stray population is in top left
if False:
    c = combined_data_labels_metadata[mask_preprocess]['celltype.l2']
    vals = np.unique(c)
    vals2num = {v: i for i, v in enumerate(sorted(vals))}
    c = c.apply(lambda tmp: vals2num[tmp])
    mask = c == 5

    fig.clf()
    ax = fig.subplots(1, 1)
    ax.scatter(e_eval_reals[:, 0], e_eval_reals[:, 1], c='darkgray', s=1, alpha=.1)
    ax.scatter(e_eval_reals[mask, 0], e_eval_reals[mask, 1], c='r', s=1)
    # scat = ax.scatter(e_eval_reals[mask, 0], e_eval_reals[mask, 1], cmap=mpl.cm.tab10, c=c[mask], s=1)
    # ax.legend(handles=scat.legend_elements()[0], labels=scat.legend_elements()[1])
    [ax.set_xticks([]), ax.set_yticks([])]
    ax.set_title('CD8+ naive')
    # fig.savefig('figures/tmp.png')

    #########################

    #########################
    # to plot sanity check genes to connect their plot in 2B to our umap

    fig.clf()
    ax = fig.subplots(1, 1)
    ax.scatter(e_eval_reals[:, 0], e_eval_reals[:, 1])
    lims_umap = [ax.get_xlim(), ax.get_ylim()]


    genes_to_plot = ['GZMB', 'MKI67', 'ITGAE', 'ZNF683', 'PDCD1', 'CTLA4', 'GZMK', 'KLF2']

    mask = np.logical_and(x_label[:, col_bloodtumor] == 1, x_label[:, col_celltype] == 1)
    r = np.random.choice(range(e_eval_reals[mask].shape[0]), e_eval_reals[mask].shape[0], replace=False)
    fig.set_size_inches([4, 8])
    fig.clf()
    axes = fig.subplots(4, 2)
    for i, ax in enumerate(axes.flatten()):
        gene_to_plot = genes_to_plot[i]
        ax.scatter(e_eval_reals[mask][r, 0], e_eval_reals[mask][r, 1], s=1, cmap=mpl.cm.bwr, c=data_rna_ambient[mask][r, gene_col_dict[gene_to_plot]])
        [ax.set_title(gene_to_plot), ax.set_xticks([]), ax.set_yticks([])]
        [ax.set_xlim(lims_umap[0]), ax.set_ylim(lims_umap[1])]
    # fig.savefig('figures/tmp.png')
    fig.set_size_inches(old_fig_size)

    #########################

    #########################
    # tcf7 / stemlike population

    thresh_tcf7 = 0.3

    # mask = x_label[:, col_celltype] == 1
    counts, bins = np.histogram(data_rna_ambient[:, gene_col_dict['TCF7']], bins=100)
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.bar(bins[1:], counts, width=bins[1] - bins[0])
    ax.axvline(thresh_tcf7, linestyle='--', c='k')
    ax.set_title('TCF7')
    # fig.savefig('figures/tmp.png')


    mask_is_stemlike = data_rna_ambient[:, gene_col_dict['TCF7']] > thresh_tcf7

    r = np.random.choice(range(e_eval_reals.shape[0]), e_eval_reals.shape[0], replace=False)
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.scatter(e_eval_reals[r, 0], e_eval_reals[r, 1], s=1, cmap=mpl.cm.bwr, c=mask_is_stemlike[r])
    [ax.set_xticks([]), ax.set_yticks([])]
    ax.set_title('Is Stemlike')
    # fig.savefig('figures/tmp.png')


    # is accuracy in stemlike population the same as not?

    pseudo_tcrs_counts = np.zeros([int(pseudo_tcrs_holdout.max() + 1), 4])
    for ii, [f1, f2] in enumerate([[0, 0], [0, 1], [1, 0], [1, 1]]):
        a, b = np.unique(pseudo_tcrs_holdout[np.logical_and(x_label[:, col_bloodtumor] == f1, x_label[:, col_prepost] == f2)], return_counts=True)
        for ai, bi in zip(a, b):
            pseudo_tcrs_counts[int(ai), ii] = bi

    def expansion_prediction_by_group_label(output_folders, num_points=10000):
        results_across_experiments_real = []
        results_across_experiments_pseudo = []

        for output_folder in output_folders:
            print(output_folder)

            #############################################
            # load args

            # output_folder_full = os.path.join(os.path.expanduser('~'), 'tcr', 'output', output_folder)
            output_folder_full = output_folder
            args = argparse.ArgumentParser().parse_args()
            with open(os.path.join(output_folder_full, 'args.txt'), 'r') as f:
                args.__dict__ = json.load(f)
            #############################################

            i_pid = int(args.heldout_patient)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            x_rna = data_rna
            x_tcr = data_tcr
            x_label = data_labels

            mask_preprocess = x_label[:, col_tcr] != ID_NULL_TCR
            x_rna = x_rna[mask_preprocess]
            x_tcr = x_tcr[mask_preprocess]
            x_label = x_label[mask_preprocess]


            G = Generator(args)
            G.load_state_dict(torch.load(os.path.join(output_folder_full, 'model.pth')))

            G.eval()
            G = G.to(device)

            #############################################
            #############################################
            # load output
            print('Starting to load output...')
            t = time.time()

            with open(os.path.join(output_folder_full, 'preds.npz'), 'rb') as f:
                npzfile = np.load(f)
                preds_rna = npzfile['preds_rna']
                preds_tcr = npzfile['preds_tcr']
                recon_rna_z = npzfile['recon_rna_z']
                recon_tcr_z = npzfile['recon_tcr_z']

                # pseudo_tcrs = npzfile['pseudo_tcrs']
                # tcr_dists = npzfile['tcr_dists']
                thresh_fitted = npzfile['thresh_fitted']


            print('Loaded output in {:.1f} s'.format(time.time() - t))

            #############################################
            #############################################
            # do evaluation


            if 0 == np.logical_and(x_label[:, col_patient] == i_pid, np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0)).sum():
                print('No BB!')
                continue
            if 0 == np.logical_and(x_label[:, col_patient] == i_pid, np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 1)).sum():
                print('No BA!')
                continue


            ########
            # get z's (mu/logvar)

            mask_pid_bb = np.logical_and(x_label[:, col_patient] == i_pid, np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0))
            recon_rna_z = numpy2torch(recon_rna_z)
            recon_tcr_z = numpy2torch(recon_tcr_z)

            mu = torch.stack([recon_rna_z[mask_pid_bb][:, :args.dimz], recon_tcr_z[mask_pid_bb][:, :args.dimz]], axis=0).mean(axis=0)
            logvar = torch.stack([recon_rna_z[mask_pid_bb][:, args.dimz:], recon_tcr_z[mask_pid_bb][:, args.dimz:]], axis=0).mean(axis=0)


            ########
            # now make predictions with those z's

            num_samples = int(num_points / mu.shape[0])
            print('num_samples: {}'.format(num_samples))

            all_out_rna = []
            all_out_tcr = []

            for _ in range(num_samples):
                out_rna_ = []
                out_tcr_ = []
                batch_random = reparameterize(mu, logvar)
                # batch_random = mu

                ones = torch.ones(batch_random.shape[0]).to(device).type(torch.int32)

                for bloodtumor, prepost in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                    batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, bloodtumor * ones)
                    batch_prepost_embeddings  = torch.index_select(G.prepost_embeddings_matrix, 0, prepost * ones)
                    batch_patient_embeddings  = torch.index_select(G.patient_embeddings_matrix, 0, i_pid * ones)

                    out_rna, out_tcr, [_, _, _] = G.sample(z=batch_random,
                                                        embeddings=[batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings])
                    out_rna_.append(out_rna.detach().cpu().numpy())
                    out_tcr_.append(out_tcr.detach().cpu().numpy())
                all_out_rna.append(out_rna_)
                all_out_tcr.append(out_tcr_)

            all_out_rna = np.concatenate(all_out_rna, axis=1)
            all_out_tcr = np.concatenate(all_out_tcr, axis=1)
            print('Got predictions')

            ########
            # get pseudoclones per condition
            all_dists = []
            for condition in range(2):

                dists = sklearn.metrics.pairwise_distances(all_out_tcr[condition], all_out_tcr[condition], metric='l1')
                all_dists.append(dists)
                print('dists {} done'.format(condition))


            all_pseudo_tcrs = []
            for condition in range(2):

                dists = all_dists[condition]

                thresh = thresh_fitted #/ 2
                pseudo_tcrs = - 10 * np.ones(all_out_tcr[condition].shape[0])
                curr_tcr_id = 0
                while (pseudo_tcrs == -10).sum() > 0:
                    if curr_tcr_id % 1000 == 0:
                        print("{:>5}: {:>5}".format(curr_tcr_id, pseudo_tcrs.shape[0] - (pseudo_tcrs != -10).sum()))
                    i = np.random.choice(np.argwhere(pseudo_tcrs == -10).flatten())

                    row_dists = dists[i]

                    mask = np.logical_and(row_dists < thresh, pseudo_tcrs == -10)
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


            all_real_clonalities = np.take(df_all_tcrs.fillna(0).values, x_label[mask_pid_bb, col_tcr].astype(int), axis=0)[:, :2]


            results_across_experiments_real.append(all_real_clonalities)
            results_across_experiments_pseudo.append(all_pseudo_clonalities)

        return results_across_experiments_real, results_across_experiments_pseudo

    ####
    output_folders = [fn for fn in sorted(glob.glob('/data/che/TRIM/HNSCC/output/holdout*'))]
    # output_folders = [o for o in output_folders if os.path.exists(os.path.join('output', o, 'preds.npz'))]

    out_real, out_pred = expansion_prediction_by_group_label(output_folders, num_points=10000)

    out_real = np.concatenate(out_real)
    out_pred = np.concatenate(out_pred)
    ####

    mask = np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0)
    mask = np.logical_and(mask, np.vectorize(lambda tmp: tmp not in [17, 20, 22])(x_label[:, col_patient]))
    clusts = combined_data_labels_metadata[mask_preprocess][mask]['celltype.l2']
    vals, counts = np.unique(clusts, return_counts=True)
    print(vals)
    print(counts)

    out_labels = []
    out_labels.append(np.ones(x_label.shape[0]).astype(bool))
    out_labels.append(mask_is_stemlike)
    out_labels.append(combined_data_labels_metadata[mask_preprocess]['celltype.l2'].apply(lambda tmp: 'Naive' in tmp).values)
    out_labels.append(combined_data_labels_metadata[mask_preprocess]['celltype.l2'].apply(lambda tmp: 'TEM' in tmp).values)
    out_labels.append(combined_data_labels_metadata[mask_preprocess]['celltype.l2'].apply(lambda tmp: 'TCM' in tmp).values)
    out_labels.append(combined_data_labels_metadata[mask_preprocess]['celltype.l2'].apply(lambda tmp: 'Treg' in tmp).values)


    confusion_matrix = np.zeros([len(out_labels), 2, 2])
    for i_out_label, out_label in enumerate(out_labels):
        out_label = out_label[mask]
        out_real_group = out_real[out_label]
        out_pred_group = out_pred[out_label]

        ytrue = out_real_group[:, 0] < out_real_group[:, 1]
        ypred = out_pred_group[:, 1] / out_pred_group[:, 0]
        ypred *= (out_pred_group[:, 0] >=2).astype(int)

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(ytrue, ypred)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        # print(roc_auc)

        thresh = thresholds[(tpr - fpr).argsort()[-1]]

        for i_tcr in range(out_real_group.shape[0]):
            true_expanded = ytrue[i_tcr].astype(int)
            pred_expanded = (ypred[i_tcr] >= thresh).astype(int)
            confusion_matrix[i_out_label, true_expanded, pred_expanded] += 1


    confusion_matrix_ratio = confusion_matrix / confusion_matrix.sum(axis=2, keepdims=True)

    confusion_matrix = np.concatenate([np.concatenate([confusion_matrix[0], confusion_matrix[1], confusion_matrix[2]], axis=-1), np.concatenate([confusion_matrix[3], confusion_matrix[4], confusion_matrix[5]], axis=-1)], axis=0)
    confusion_matrix_ratio = np.concatenate([np.concatenate([confusion_matrix_ratio[0], confusion_matrix_ratio[1], confusion_matrix_ratio[2]], axis=-1), np.concatenate([confusion_matrix_ratio[3], confusion_matrix_ratio[4], confusion_matrix_ratio[5]], axis=-1)], axis=0)

    fig.clf()
    ax = fig.subplots(1, 1)
    sns.heatmap(confusion_matrix_ratio, ax=ax, cmap=mpl.cm.bwr)
    for i, i_tick in enumerate(ax.get_xticks()):
        for j, j_tick in enumerate(ax.get_yticks()):
            ax.text(i_tick, j_tick, int(confusion_matrix[j, i]), ha='center')
    ax.axhline(2, linewidth=5, c='k')
    ax.axvline(2, linewidth=5, c='k')
    ax.axvline(4, linewidth=5, c='k')
    ax.set_xticklabels(['Pred\nNot-expand', 'Pred\nExpand', 'Pred\nNot-expand', 'Pred\nExpand', 'Pred\nNot-expand', 'Pred\nExpand'], fontsize=6)
    ax.set_yticklabels(['True\nNot-expand', 'True\nExpand', 'True\nNot-expand', 'True\nExpand'], rotation=0, fontsize=6)
    ax.set_title('Expansion prediction confusion matrix for cell subsets:\nAll, Stem, Naive, TEM, TCM, Treg')
    # fig.savefig('figures/tmp.png')




    #########################
    # now explore expansion in stemlike cells per patient

    'any_resp'
    'near_complete_resp'
    'volum_resp'
    'path_downstage',

    patient_ids = ['P23', 'P24', 'P29', 'P32', 'P01', 'P02', 'P04', 'P05', 'P08', 'P09', 'P10', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P25', 'P26', 'P28',  'P30', 'P31']
    counts1 = pd.crosstab(df_blood_metadata['patient'], df_blood_metadata['bloodid'])
    counts2 = pd.crosstab(df_tumor_metadata['patient'], df_tumor_metadata['pre_post'])
    counts = pd.concat([counts1, counts2], axis=1).fillna(0).astype(int)[['B1', 'B2', 'pre', 'post']]
    counts = counts.loc[patient_ids]
    bad_pids = np.argwhere(np.logical_or(counts.iloc[:, 0] == 0, counts.iloc[:, 1] == 0).values).flatten().tolist()
    # mask_is_stemlike = data_rna_ambient[:, gene_col_dict['TCF7']] > .2



    mask = np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0)
    mask = np.logical_and(mask, np.vectorize(lambda tmp: tmp not in bad_pids)(x_label[:, col_patient]))
    tmp_tcr_counts = np.take(df_all_tcrs, x_label[mask, col_tcr], axis=0).fillna(0).values[:, :2] 
    # tmp_tcr_counts = out_real


    p_matrix = pd.crosstab(df_blood_metadata['volum_resp'], df_blood_metadata['patient']).T.loc[patient_ids].values
    if p_matrix.shape[1] == 3:
        p_matrix = p_matrix[:, 1:]
    responders = p_matrix.argmax(-1)


    results = []
    for i_pid in np.unique(x_label[mask][:, col_patient]):
        mask_patient = np.logical_and(x_label[mask][:, col_patient] == i_pid, mask_is_stemlike[mask])
        num_expanded = (tmp_tcr_counts[mask_patient][:, 0] < tmp_tcr_counts[mask_patient][:, 1]).sum()
        num_total = mask_patient.sum()

        i_pid = int(i_pid)
        results.append([responders[i_pid], 1. * num_expanded / num_total])

        color = 'green' if responders[i_pid] else 'red'
        print(termcolor.colored("{:>2}:  {:.2f}%  ({:>4} / {:>4})   {}".format(int(i_pid), 1. * num_expanded / num_total, num_expanded, num_total, responders[i_pid]), color))
    results = np.array(results)


    print('{:.3f}'.format(np.corrcoef(results[:, 0], results[:, 1])[0, 1]))
    print(np.median(results[results[:, 0] == 0][:, 1]))
    print(np.median(results[results[:, 0] == 1][:, 1]))

    fig.clf()
    ax = fig.subplots(1, 1)
    ax.scatter(results[:, 0], results[:, 1], c='k', s=5)
    ax.set_xticks([0, 1])
    ax.set_xlim([-.5, 1.5])
    ax.plot([-.15, .15], 2 * [np.median(results[results[:, 0] == 0][:, 1])], c='k', linestyle='--')
    ax.plot([.85, 1.15], 2 * [np.median(results[results[:, 0] == 1][:, 1])], c='k', linestyle='--')
    ax.set_xticklabels(['Nonresponders', 'Responders'])
    ax.set_ylabel('% BB stemlike cells that expand in BA')
    ax.annotate('r = {:.2f}'.format(np.corrcoef(results[:, 0], results[:, 1])[0, 1]), xy=[.8, .94], xycoords='axes fraction', fontsize=14)
    ax.set_title('path_downstage')
    # fig.savefig('figures/tmp.png')





    #########################
    # to plot clonality count label on TCR space

    mask = get_drop_duplicates_mask(x_tcr)

    umapper = umap.UMAP()
    e = umapper.fit_transform(x_tcr[mask])

    c = np.take(df_all_tcrs.fillna(0).sum(axis=1).values, x_label[mask][:, col_tcr].astype(int), axis=0)
    fig.clf()
    ax = fig.subplots(1, 1)
    r = np.random.choice(range(e.shape[0]), e.shape[0], replace=False)
    ax.scatter(e[r, 0], e[r, 1], s=1, c=c[r], vmax=5)
    [ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
    # fig.savefig('figures/tmp.png')


    #########################




    #########################
    # are there individual genes that are correlated to TCR distance?

    data_rna_raw_filtered = data_rna_raw[mask_preprocess]


    num_samples = 10
    num_cells_per_sample = 1000
    num_genes = 29750

    corrs_all = []
    for num_sample in range(num_samples):
        r = np.random.choice(range(data_rna_raw_filtered.shape[0]), num_cells_per_sample, replace=False)

        tmp2 = x_tcr[r]
        dists2 = sklearn.metrics.pairwise_distances(tmp2, tmp2)

        corrs_round = []
        for col in range(num_genes):
            if col % 10000 == 0: print(col)

            tmp1 = data_rna_raw_filtered[r, col].reshape([-1, 1])
            
            if (tmp1 == 0).all():
                corrs_round.append(0)
                continue

            dists1 = sklearn.metrics.pairwise_distances(tmp1, tmp1)
            
            mask_triu = (np.triu(np.ones([dists1.shape[0], dists1.shape[0]])) - np.eye(dists1.shape[0])).flatten() == 1
            corr = np.corrcoef(dists1.flatten()[mask_triu], dists2.flatten()[mask_triu])[0, 1]
            corrs_round.append(corr)
        corrs_round = np.array(corrs_round)
        corrs_all.append(corrs_round)
        print("round {}, max |r| = {:.3f}".format(num_sample, np.abs(np.array(corrs_all).mean(axis=0)).max()))

        # corrs_all_mean = np.array(corrs_all).mean(axis=0)
        # print("# {:>6.3f}  {}".format(corrs_all_mean.min(), combined_data_columns[corrs_all_mean.argmin()]))
        # print("# {:>6.3f}  {}".format(corrs_all_mean.max(), combined_data_columns[corrs_all_mean.argmax()]))

        corrs_all_array = np.array(corrs_all)
        corrs_all_mean = corrs_all_array.mean(axis=0)

        for i in range(15):
            print("{:<2}  {:<14}  ( {:>7.3f} )".format(i, combined_data_columns[corrs_all_mean.argsort()[::-1][i]], np.sort(corrs_all_mean)[::-1][i]))


    # no gene is more correlated than:
    # round 9, max |r| = 0.034
    # 0   TRBV20-1        (   0.235 )
    # 1   TRBV29-1        (   0.142 )
    # 2   TRBV30          (   0.102 )
    # 3   TRBV10-3        (   0.060 )
    # 4   TRBV12-5        (   0.036 )
    # 5   TRBV6-2         (   0.035 )
    # 6   TRBV15          (   0.026 )
    # 7   TRAV14DV4       (   0.025 )
    # 8   TRAV19          (   0.024 )
    # 9   TRBV24-1        (   0.022 )
    # 10  AC106801.1      (   0.022 )
    # 11  AC011921.1      (   0.022 )
    # 12  DHDH            (   0.021 )
    # 13  MS4A4E          (   0.019 )
    # 14  AC004130.2      (   0.019 )
    #########################

    corr_gene_names = ['TRBV20-1', 'TRBV29-1', 'TRBV30', 'TRBV10-3', 'TRBV12-5', 'TRBV6-2', 'TRBV15', 'TRAV14DV4', 'TRAV19', 'TRBV24-1', 'AC106801.1', 'AC011921.1', 'DHDH']
    corr_gene_values = [.235, .142, .102, .060, .036, .035, .026, .025, .024, .022, .022, .022, .021]
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.barh(range(len(corr_gene_values)), corr_gene_values[::-1])
    ax.set_yticks(range(len(corr_gene_names)))
    ax.set_yticklabels(corr_gene_names[::-1], rotation=0, fontsize=7)
    # fig.savefig('figures/tmp.png')

    #########################





    #########################
    # does generated data match training data?

    umapper = umap.UMAP()

    e_real = umapper.fit_transform(data_rna[mask_preprocess])
    e_pred = umapper.transform(preds_rna_holdout)



    fig.subplots_adjust(.05, .05, .95, .95)
    fig.set_size_inches([8, 8])
    fig.clf()
    axes = fig.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            ax = axes.flatten()[2 * i + j]
            mask = np.logical_and(x_label[:, col_bloodtumor] == i, x_label[:, col_prepost] == j)

            tmp_plot = np.concatenate([e_real[mask], e_pred[mask]], axis=0)
            tmp_plot_l = np.concatenate([np.zeros(mask.sum()), np.ones(mask.sum())])
            r = np.random.choice(range(tmp_plot.shape[0]), tmp_plot.shape[0], replace=False)
            ax.scatter(tmp_plot[r, 0], tmp_plot[r, 1], s=1, c=tmp_plot_l[r], alpha=.75)
            [ax.set_xticks([]), ax.set_yticks([])]
    # fig.savefig('figures/tmp.png')
    fig.set_size_inches(old_fig_size)

    #########################


    #########################
    # histogram of clonotype sizes in cd4/cd8, blood/tumor subsets



    # clone_count_by_cell_real = np.take(df_all_tcrs.fillna(0).sum(axis=1).values, x_label[:, col_tcr].astype(np.int32))

    mask = np.logical_and(x_label[:, col_bloodtumor] == 1, x_label[:, col_celltype] == 0)
    tcrs_in_mask = np.unique(x_label[mask, col_tcr])
    tcrs_in_mask_counts = np.take(df_all_tcrs.fillna(0).sum(axis=1).values, tcrs_in_mask.astype(np.int32))

    counts, bins = np.unique(tcrs_in_mask_counts, return_counts=True)

    counts_plus = counts[bins >= 50].sum()
    counts = np.concatenate([counts[bins < 50], [counts_plus]], axis=-1)
    bins = np.concatenate([bins[bins < 50], [50]], axis=-1)

    fig.clf()
    ax = fig.subplots(1, 1)
    ax.bar(bins, counts, width=1)
    # fig.savefig('figures/tmp.png')
    #########################




