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
fig = plt.figure()



def get_args():
    parser = argparse.ArgumentParser()

    # model ID
    parser.add_argument('--model_name', default='holdout0')
    # parser.add_argument('--output_folder', default='output/test')
    parser.add_argument('--output_folder', default='/data/che/TRIM/HNSCC/output')
    parser.add_argument('--heldout_patient', default=0)

    # data location
    # parser.add_argument('--processed_data_folder', default='data_processed')
    parser.add_argument('--processed_data_folder', default='/home/che/TRIM/data_processed')
    parser.add_argument('--umap_trained_file', default='umap_trained.pkl')

    # training
    parser.add_argument('--training_steps', default=20000)
    parser.add_argument('--lr', default=.001)
    parser.add_argument('--print_every', default=100)
    parser.add_argument('--save_every', default=10000)

    # model architecture
    parser.add_argument('--learn_patient_embeddings', default=1)
    parser.add_argument('--reduction', default='PCA', choices=['PCA', 'HVG'])
    parser.add_argument('--delta_contrastive', default=10)

    # model size
    parser.add_argument('--n_learn_embedding_sample', default=5120, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--n_channels_base', default=2048, type=int)
    parser.add_argument('--dimz', default=1024)
    parser.add_argument('--dim_state_embedding', default=128)
    
    # loss terms
    parser.add_argument('--lambda_kl', default=15)
    parser.add_argument('--lambda_recon_rna', default=1)
    parser.add_argument('--lambda_recon_tcr', default=1)
    parser.add_argument('--lambda_embedding_norm', default=10)

    args = parser.parse_args()

    args.i_iter = 0

    return args

args = get_args()
args.output_folder = os.path.join(args.output_folder, 'holdout{}'.format(args.heldout_patient))
if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
print('Output folder: {}'.format(args.output_folder))
print('Batch size: {}'.format(args.batch_size))

###############
############### START data loading
############### 
print('Starting to load data...')
t = time.time()

with open(os.path.join(args.processed_data_folder, 'data_rna.npz'), 'rb') as f:
    npzfile = np.load(f)
    data_rna = npzfile['data_rna']

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




col_bloodtumor, col_prepost, col_celltype, col_patient, col_tcr, col_tcr_v, col_tcr_j, col_treatment = list(range(data_labels.shape[1]))
ID_NULL_TCR = 0

print('Loaded data in {:.1f} s'.format(time.time() - t))
###############
############### END data loading
############### 


###############
############### START train model
############### 
print('Starting to pca/viz_umap data...')
t = time.time()

if args.reduction == 'PCA':
    npca = 100
    pca = sklearn.decomposition.PCA(npca, random_state=0)
    data_rna = pca.fit_transform(data_rna)

elif args.reduction == 'HVG':
    nhvg = 1000
    mask_hvg = scanpy.pp.highly_variable_genes(scanpy.AnnData(data_rna, data_rna.index.to_frame(), data_rna.columns.to_frame()), inplace=False, n_top_genes=nhvg)['highly_variable'].values


if not os.path.exists(args.umap_trained_file):
    viz_reducer = umap.UMAP(random_state=0).fit(data_rna)
    pickle.dump(viz_reducer, open(args.umap_trained_file, 'wb+'))
    print('Trained UMAP and saved to {}'.format(args.umap_trained_file))
else:
    viz_reducer = pickle.load(open(args.umap_trained_file, 'rb'))
    print('Loaded UMAP from {}'.format(args.umap_trained_file))

e_eval_reals = viz_reducer.transform(data_rna)


print('Finished pca/viz_umap data in {:.1f} s'.format(time.time() - t))

#  use gpu if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


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

# def cluster_logos(tcrbert_embeddings, df_all_tcrs, clustering_method='kmeans', n_clusts=5):
#     if clustering_method == 'kmeans':
#         km = sklearn.cluster.KMeans(n_clusts)
#         clusts_ = km.fit_predict(tcrbert_embeddings)
#     elif clustering_method == 'louvain':
#         clusts_, _, _ = phenograph.cluster(tcrbert_embeddings)
#     else:
#         raise Exception("bad clustering method")


#     n_clusts = len(np.unique(clusts_))

#     umapper = umap.UMAP()
#     e = umapper.fit_transform(tcrbert_embeddings)


#     r = np.random.choice(range(e.shape[0]), e.shape[0], replace=False)
#     fig.set_size_inches(6, 6)
#     fig.clf()
#     ax = fig.subplots(1, 1)
#     # make_legend(ax, ['0', '1', '2', '3', '4'], cmap=mpl.cm.viridis)
#     ax.scatter(e[r, 0], e[r, 1], s=1, cmap=mpl.cm.tab20, c=clusts_[r])
#     [[ax.set_xticks([]), ax.set_yticks([])] for ax in [ax]]
#     [ax.set_xlabel('UMAP1'), ax.set_ylabel('UMAP2')]
#     fig.savefig("tmp_tcr_umap.png")


#     fig.set_size_inches(20, 5)
#     for i in range(len(np.unique(clusts_))):
#         fig.clf()
#         ax = fig.subplots(1, 1)
#         tmp = df_all_tcrs[clusts_ == i]
#         tmp = [tmp.replace(' ', '').ljust(22) for tmp in tmp.index if tmp.replace(' ', '')]

#         counts = lm.alignment_to_matrix(tmp)
#         lm.Logo(counts, ax=ax, color_scheme='chemistry')

#         fig.savefig('tmp_logo{}.png'.format(i))

def clear_torch_memory():
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass
    torch.cuda.empty_cache()

def make_legend(ax, labels, s=20, cmap=mpl.cm.jet, **kwargs):
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

def cross_entropy(x, y):
    """ Computes cross entropy between two distributions.
    Input: x: iterabale of N non-negative values
           y: iterabale of N non-negative values
    Returns: scalar
    """

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError('Negative values exist.')

    # Force to proper probability mass function.
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x /= np.sum(x)
    y /= np.sum(y)

    # Ignore zero 'y' elements.
    mask = y > 0
    x = x[mask]
    y = y[mask]    
    ce = -np.sum(x * np.log(y)) 
    return ce

def check_celltype_accuracy(preds, real, labels):
    # classifier = sklearn.svm.SVC()
    # classifier.fit(real, labels[:, 2])

    # preds_labels = classifier.predict(preds)

    # _, real_counts = np.unique(labels[:, 2], return_counts=True)
    # _, preds_labels_counts = np.unique(preds_labels, return_counts=True)

    # print(real_counts)
    # print(preds_labels_counts)
    # print(np.round(real_counts / real_counts.sum(), 2))
    # print(np.round(preds_labels_counts / preds_labels_counts.sum(), 2))

    

    for i_pid in range(len(patient_ids)):
        print('Patient {}'.format(i_pid))
        ce_s = []
        mmd_s = []

        mask1 = np.logical_and(np.logical_and(labels[:, 0] == 0, labels[:, 1] == 0), labels[:, 3] == i_pid)
        mask2 = np.logical_and(np.logical_and(labels[:, 0] == 0, labels[:, 1] == 1), labels[:, 3] == i_pid)
        mask3 = np.logical_and(np.logical_and(labels[:, 0] == 1, labels[:, 1] == 0), labels[:, 3] == i_pid)
        mask4 = np.logical_and(np.logical_and(labels[:, 0] == 1, labels[:, 1] == 1), labels[:, 3] == i_pid)

        for i_mask, mask in enumerate([mask1, mask2, mask3, mask4]):
            print("mask {}".format(i_mask))
            if sum(mask) == 0: continue

            classifier = sklearn.svm.SVC()
            classifier.fit(real[mask, :], labels[mask, 2])
            preds_subset = classifier.predict(preds[mask])

            real_subset = labels[mask]

            real_counts = [(real_subset[:, 2] == 0).sum(), (real_subset[:, 2] == 1).sum(), (real_subset[:, 2] == 2).sum()]
            preds_labels_counts = [sum(preds_subset == 0), sum(preds_subset == 1), sum(preds_subset == 2)]

            # print(real_counts)
            # print(preds_labels_counts)
            real_counts = real_counts / sum(real_counts)
            preds_labels_counts = preds_labels_counts / sum(preds_labels_counts)
            print('    {}'.format(np.round(real_counts, 2)))
            print('    {}'.format(np.round(preds_labels_counts, 2)))

            ce = cross_entropy(real_counts, preds_labels_counts)
            ce_s.append(ce)
            print('    CE: {:.3f}'.format(ce))
            
            mmd = mmd_polynomial(real[mask], preds[mask])
            mmd_s.append(mmd)
            print('    MMD: {:.3f}'.format(mmd))


        print('    Mean CE: {:.3f}'.format(np.mean(ce_s)))
        print('    Mean MMD: {:.3f}'.format(np.mean(mmd_s)))

def mmd_polynomial(X, Y, gammas=[.01, .1, 1, 10, 100]):
    mmd = 0.
    for gamma in gammas:
        XX = sklearn.metrics.pairwise.polynomial_kernel(X, X, 2, gamma)
        YY = sklearn.metrics.pairwise.polynomial_kernel(Y, Y, 2, gamma)
        XY = sklearn.metrics.pairwise.polynomial_kernel(X, Y, 2, gamma)
        mmd += XX.mean() + YY.mean() - 2 * XY.mean()

    mmd = mmd / len(gammas)

    mmd /= 10e6

    return mmd

# def cell_type_plot(embeddings, labels):
#     fig.clf()
#     ax = fig.subplots(1, 1)
#     make_legend(ax, ['CD4', 'CD8', 'Myeloid'], cmap=mpl.cm.copper)
#     r = np.random.choice(embeddings.shape[0], embeddings.shape[0], replace=False)
#     ax.scatter(embeddings[r, 0], embeddings[r, 1], c=labels[r], s=1, cmap=mpl.cm.copper)
#     [ax.set_xticks([]), ax.set_yticks([])]
#     ax.set_xlabel('UMAP1')
#     ax.set_ylabel('UMAP2')
#     fig.savefig('tmp_celltype.png')

def replace_with_zeros(mask, data):
    null_vec = torch.zeros([1, data.shape[1]]).repeat(data.shape[0], 1).to(device)

    mask = mask[:,np.newaxis].repeat(1, data.shape[1])

    out = torch.where(mask, null_vec, data)

    return out

def numpy2torch(x, type=torch.FloatTensor):

    return torch.from_numpy(x).type(type).to(device)

def one_hot_along_3rd_axis(x):
    out = np.zeros([x.shape[0], x.shape[1], len(vocab)])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j, x[i, j]] = 1
    return out

def unique_sorted_by_count(x, random=True):
    vals, counts = np.unique(x, return_counts=True, axis=0)
    if random:
        chosen_ind = np.random.choice(list(range(counts.shape[0])), 1, p=counts / counts.sum())
        return vals[chosen_ind]
    else:
        vals = vals[np.argsort(-counts)]
        return vals[0]

def get_atchley():
    atchley = pd.DataFrame(
            [[' ', 0, 0, 0, 0, 0],
            ['A', -0.59145974, -1.30209266, -0.7330651,  1.5703918, -0.14550842],
            ['C', -1.34267179,  0.46542300, -0.8620345, -1.0200786, -0.25516894],
            ['D',  1.05015062,  0.30242411, -3.6559147, -0.2590236, -3.24176791],
            ['E',  1.35733226, -1.45275578,  1.4766610,  0.1129444, -0.83715681],
            ['F', -1.00610084, -0.59046634,  1.8909687, -0.3966186,  0.41194139],
            ['G', -0.38387987,  1.65201497,  1.3301017,  1.0449765,  2.06385566],
            ['H',  0.33616543, -0.41662780, -1.6733690, -1.4738898, -0.07772917],
            ['I', -1.23936304, -0.54652238,  2.1314349,  0.3931618,  0.81630366],
            ['K',  1.83146558, -0.56109831,  0.5332237, -0.2771101,  1.64762794],
            ['L', -1.01895162, -0.98693471, -1.5046185,  1.2658296, -0.91181195],
            ['M', -0.66312569, -1.52353917,  2.2194787, -1.0047207,  1.21181214],
            ['N',  0.94535614,  0.82846219,  1.2991286, -0.1688162,  0.93339498],
            ['P',  0.18862522,  2.08084151, -1.6283286,  0.4207004, -1.39177378],
            ['Q',  0.93056541, -0.17926549, -3.0048731, -0.5025910, -1.85303476],
            ['R',  1.53754853, -0.05472897,  1.5021086,  0.4403185,  2.89744417],
            ['S', -0.22788299,  1.39869991, -4.7596375,  0.6701745, -2.64747356],
            ['T', -0.03181782,  0.32571153,  2.2134612,  0.9078985,  1.31337035],
            ['V', -1.33661279, -0.27854634, -0.5440132,  1.2419935, -1.26225362],
            ['W', -0.59533918,  0.00907760,  0.6719274, -2.1275244, -0.18358096],
            ['Y',  0.25999617,  0.82992312,  3.0973596, -0.8380164,  1.51150958]])
    atchley = atchley.set_index(0)
    return atchley

def reparameterize(mu, logvar, clamp=5):
    std = logvar.mul(0.5).exp_()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)

    # std = torch.exp(0.5 * logvar) # standard deviation
    # eps = torch.randn_like(std) # `randn_like` as we need the same size
    # sample = mu + (eps * std) # sampling as if coming from the input space
    # return sample

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            # bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            bandwidth = torch.mean(L2_distance.data.detach())
        bandwidth /= kernel_mul ** (kernel_num // 2)

        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        # loss = torch.mean(XX + YY - XY - YX)
        loss = XX.mean() + YY.mean() - XY.mean() - YX.mean()
        return loss

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
        self.register_buffer('prepost_embeddings_matrix', torch.zeros(2, args.dim_state_embedding))
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

class CNN_AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.nbase = nbase = kwargs['nbase']
        self.dim_in = dim_in = kwargs["dim_in"][1]
        self.dimz = dimz = kwargs['dimz']
        self.dim_len = tcr_max_len

        # TCR
        # total_dim_in_tcr = dim_in[1]
        # self.encoder_tcr = CNN(dim_in=total_dim_in_tcr, dim_out=dimz * 6, dim_len=tcr_max_len, nbase=nbase * 1)
        # self.decoder_tcr = CNN_T(dim_in=dimz * 6, dim_out=dim_in[1], dim_len=tcr_max_len, nbase=nbase * 1)

        ksize = 5
        self.conv1 = torch.nn.Conv2d(1, dimz // 1, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2)
        self.conv2 = torch.nn.Conv2d(dimz // 1, dimz // 2, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2)

        self.conv1_T = torch.nn.ConvTranspose2d(dimz // 2, dimz // 1, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2, output_padding=1)
        self.conv2_T = torch.nn.ConvTranspose2d(dimz // 1, 1, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2, output_padding=1)
        
        self.bn1 = nn.BatchNorm2d(dimz // 1)
        self.bn2 = nn.BatchNorm2d(dimz // 2)
        self.bn1_T =  nn.BatchNorm2d(dimz // 1)
        self.bn2_T =  nn.BatchNorm2d(dimz // 1)

        self.z_fc1 = nn.Linear(in_features=dimz * 20, out_features=dimz)
        self.z_fc2 = nn.Linear(in_features=dimz, out_features=dimz)
        self.z_fc3 = nn.Linear(in_features=dimz, out_features=dimz)
        self.z_fc4 = nn.Linear(in_features=dimz, out_features=dimz * 20)

        self.lookup_bloodtumor = self.lookup_prepost = self.lookup_treatment = lambda tmp: None

        # ops
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x, embeddings, mask_notnulltcr=None):
        x_tcr = x[1][:, np.newaxis, :, :]

        tcr_shape_in = x_tcr.shape
        h1 = self.lrelu(self.bn1(self.conv1(x_tcr)))
        h2 = self.lrelu(self.bn2(self.conv2(h1)))
        pre_flattened_shape = h2.shape
        h2 = h2.view([pre_flattened_shape[0], -1])

        z1 = self.lrelu(self.z_fc1(h2))
        z2 = self.lrelu(self.z_fc2(z1))
        final_z = z2
        z3 = self.lrelu(self.z_fc3(z2))
        z4 = self.lrelu(self.z_fc4(z3))

        d0 = z4.view(pre_flattened_shape)
        d1 = self.lrelu(self.bn1_T(self.conv1_T(d0)))
        d2 = self.conv2_T(d1)

        recon_tcr = torch.squeeze(d2[:, :, :tcr_shape_in[2], :tcr_shape_in[3]], 1)

        return None, recon_tcr, [None, None, final_z, None]

    ###### for TESSA style:
    # def __init__(self, **kwargs):
        # super().__init__()
        # self.nbase = nbase = kwargs['nbase']
        # self.dim_in = dim_in = kwargs["dim_in"][1]
        # self.dimz = dimz = kwargs['dimz']
        # self.dim_len = tcr_max_len

        # self.conv1 = torch.nn.Conv2d(1, 30, kernel_size=[5, 2], stride=1)
        # self.conv2 = torch.nn.Conv2d(30, 20, kernel_size=[4, 2], stride=1)
        
        # self.z_fc1 = nn.Linear(in_features=240, out_features=30)
        # self.z_fc2 = nn.Linear(in_features=30, out_features=30)
        # self.z_fc3 = nn.Linear(in_features=30, out_features=30)
        # self.z_fc4 = nn.Linear(in_features=30, out_features=240)
        # self.conv1_T = torch.nn.Conv2d(20, 30, kernel_size=[4, 3], stride=1)
        # self.conv2_T = torch.nn.Conv2d(30, 1, kernel_size=[6, 4], stride=1)


        # self.bn1 = nn.BatchNorm2d(30)
        # self.bn2 = nn.BatchNorm2d(20)
        # self.bn0_T = nn.BatchNorm2d(20)
        # self.bn1_T = nn.BatchNorm2d(30)

        # self.lookup_bloodtumor = self.lookup_prepost = self.lookup_treatment = lambda tmp: None

        # # ops
        # self.lrelu = torch.nn.SELU()
        # self.dropout = torch.nn.Dropout(.01)
        # self.pool = nn.AvgPool2d([4, 1])

    # def forward(self, x, embeddings, mask_notnulltcr=None):
        # x_tcr = x[1][:, np.newaxis, :, :]

        # tcr_shape_in = x_tcr.shape
        # h1 = self.lrelu(self.bn1(self.conv1(x_tcr)))
        # h1 = self.pool(h1)
        # h2 = self.lrelu(self.bn2(self.conv2(h1)))
        # h2 = self.pool(h2)
        # pre_flattened_shape = h2.shape
        # h2 = h2.view([pre_flattened_shape[0], -1])

        # z1 = self.lrelu(self.z_fc1(h2))
        # z1 = self.dropout(z1)
        # final_z = z2 = self.lrelu(self.z_fc2(z1))
        # z2 = self.dropout(z2)
        # z3 = self.lrelu(self.z_fc3(z2))
        # z3 = self.dropout(z3)
        # z4 = self.lrelu(self.z_fc4(z3))

        # d0 = z4.view(pre_flattened_shape)
        # d0 = self.bn0_T(d0)
        # d0 = nn.Upsample([20, 6])((d0))
        # d1 = self.lrelu(self.bn1_T(self.conv1_T(d0)))
        # d1 = nn.Upsample([85, 8])(d1)
        # d2 = self.conv2_T(d1)
        # recon_tcr = torch.squeeze(d2, 1)

        # # recon_tcr = torch.squeeze(d2[:, :, :tcr_shape_in[2], :tcr_shape_in[3]], 1)

        # return None, recon_tcr, [None, None, final_z, None]

class CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hdim = kwargs['nbase']
        dim_in = kwargs['dim_in']
        dim_out = kwargs['dim_out']
        dim_len = kwargs['dim_len']

        strides = [2, 2, 2]

        ksize = 3
        self.conv1 = torch.nn.Conv1d(dim_in, hdim // 1, kernel_size=ksize, stride=strides[0], padding=(ksize - 1) // 2)
        self.conv2 = torch.nn.Conv1d(hdim // 1, hdim // 2, kernel_size=ksize, stride=strides[1], padding=(ksize - 1) // 2)
        self.conv3 = torch.nn.Conv1d(hdim // 2, hdim // 4, kernel_size=ksize, stride=strides[2], padding=(ksize - 1) // 2)

        self.bn1 = nn.BatchNorm1d(hdim // 1) # nn.Identity() # 
        self.bn2 = nn.BatchNorm1d(hdim // 2) # nn.Identity() # 
        self.bn3 = nn.BatchNorm1d(hdim // 4) # nn.Identity() # 

        sum_of_strides = sum([s > 1 for s in strides])
        self.fc_out = nn.Linear(in_features=(hdim // 4) * math.ceil(dim_len / (2**sum_of_strides)), out_features=dim_out)

        self.lrelu = torch.nn.LeakyReLU()

        self.first = True

    def forward(self, x):
        x = x.permute([0, 2, 1])

        h1 = self.lrelu(self.bn1(self.conv1(x)))
        h2 = self.lrelu(self.bn2(self.conv2(h1)))
        h3 = self.lrelu(self.bn3(self.conv3(h2)))
        h3_flat = h3.view([x.shape[0], -1])

        out = self.fc_out(h3_flat)

        if self.first:
            print(h1.shape)
            print(h2.shape)
            print(h3.shape)
            print(h3_flat.shape)
            print(out.shape)
            self.first = False

        return out

class CNN_T(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hdim = self.hdim = kwargs['nbase']
        dim_in = self.dim_in = kwargs['dim_in']
        dim_out = self.dim_out= kwargs['dim_out']
        dim_len = self.dim_len = kwargs['dim_len']

        self.dim_first_len = (dim_len // 4 + 1)
        self.dim_reshape = (hdim // 4) * self.dim_first_len

        self.fc_in1 = nn.Linear(in_features=dim_in, out_features=self.dim_reshape)
        # self.conv1 = torch.nn.Conv1d(hdim // 8, hdim // 4, kernel_size=7, dilation=1, stride=1, padding='same', padding_mode='replicate')
        # self.conv2 = torch.nn.Conv1d(hdim // 4, hdim // 2, kernel_size=7, dilation=1, stride=1, padding='same', padding_mode='replicate')
        self.fc_out1 = nn.Linear(in_features=hdim // 1, out_features=dim_out)


        ksize = 3
        self.conv1 = torch.nn.ConvTranspose1d(hdim // 4, hdim // 2, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose1d(hdim // 2, hdim // 1, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2, output_padding=1)
        # self.conv_out = torch.nn.ConvTranspose1d(hdim // 1, dim_out, kernel_size=1)
        

        self.bn_in1 =  nn.BatchNorm1d(self.dim_reshape)
        self.bn1 =  nn.BatchNorm1d(hdim // 2)
        self.bn2 =  nn.BatchNorm1d(hdim // 1)
        self.bn_out =  nn.BatchNorm1d(dim_out)

        self.up =  torch.nn.Upsample(scale_factor=2)
        self.lrelu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        h1 = self.lrelu(self.bn_in1(self.fc_in1(x)))
        h1 = h1.view([x.shape[0], self.hdim // 4, self.dim_first_len])

        h2 = self.lrelu(self.bn1(self.conv1(h1)))
        h3 = self.lrelu(self.bn2(self.conv2(h2)))
        out = h3

        out = out[:, : ,:self.dim_len]
        out = out.permute([0, 2, 1])

        out = self.fc_out1(out)

        return out

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

def vectorized_choice(p, n=1, items=None):
    s = p.cumsum(axis=1)
    r = np.random.rand(p.shape[0], n, 1)
    q = np.expand_dims(s, 1) >= r
    k = q.argmax(axis=-1)
    if items is not None:
        k = np.asarray(items)[k]
    return k

def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

def check_for_nan(model):
    for name, param in model.named_parameters():
        flag = False
        if param.grad is not None and torch.isnan(param.grad).any():
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(name)
            flag = True
        if flag:
            raise Exception('My NaN error')

def get_x_tcr_from_label(labels_matrix):
    # x_tcr = one_hot_along_3rd_axis(np.take(df_all_tcrs_array, labels_matrix[:, col_tcr].astype(np.int32), axis=0))
    # x_tcr = np.take(tcrbert_embeddings, labels_matrix[:, col_tcr].astype(np.int32), axis=0)
    x_tcr = np.take(tcr_embeddings, labels_matrix[:, col_tcr].astype(np.int32), axis=0)
    # x_tcr = np.take(get_atchley().values, np.take(df_all_tcrs_array, labels_matrix[:, col_tcr].astype(np.int32), axis=0), axis=0)

    return x_tcr

def get_patient_embeddings_from_label(labels_matrix):

    return np.take(x_patient_embeddings, labels_matrix[:, col_patient].astype(np.int32) , axis=0)

def mean_by_tcr_clone(data, labels):
    tmpdf = pd.DataFrame(np.concatenate([data, labels], axis=-1), columns=list(range(data.shape[1] + labels.shape[1])))
    tmpdf = tmpdf.groupby(data.shape[1] + col_tcr).mean()
    tmpdf[data.shape[1] + col_tcr] = tmpdf.index
    tmpdf = tmpdf[sorted(tmpdf.columns)].values
    data = tmpdf[:, :data.shape[1]]

    return data, labels

def get_pseudoclones(preds_tcr, x_label, train_mask, dists, thresh, tol=.025, max_tries=50, print_every=False):
    t = time.time()

    num_unique = len(np.unique(x_label[train_mask, col_tcr]))

    good = False
    tries = 0

    while not good:
        pseudo_tcrs = - 10 * np.ones(x_label.shape[0])
        curr_tcr_id = 0
        while (pseudo_tcrs == -10).sum() > 0:
            if print_every and curr_tcr_id % print_every == 0:
                print("{:>5}: {:>5}".format(curr_tcr_id, pseudo_tcrs.shape[0] - (pseudo_tcrs != -10).sum()))
            i = np.random.choice(np.argwhere(pseudo_tcrs == -10).flatten())
            row_dists = dists[i]

            mask = np.logical_and(row_dists < thresh, pseudo_tcrs == -10)
            mask = np.logical_and(mask, x_label[:, col_bloodtumor] == x_label[i, col_bloodtumor])
            mask = np.logical_and(mask, x_label[:, col_patient] == x_label[i, col_patient])
            # mask = np.logical_and(mask, x_label[:, col_prepost] == x_label[i, col_prepost])
            pseudo_tcrs[mask] = curr_tcr_id

            curr_tcr_id += 1

        num_unique_preds = len(np.unique(pseudo_tcrs[train_mask]))

        if ((1 - tol) * num_unique) <= num_unique_preds and num_unique_preds <= ((1 + tol) * num_unique):
            good = True
        else:
            if num_unique_preds < num_unique:
                thresh *= .95
            else:
                thresh *= 1.05
        tries += 1
        if tries >= max_tries:
            break

    print("Found {} pseudo_tcrs compared to {} real_tcrs with thresh {:.2f} on {} tries ({:.1f} s)".format(num_unique_preds, num_unique, thresh, tries, time.time() - t))

    return pseudo_tcrs, thresh

def check_pseudoclones_per_condition(pseudo_tcrs, x_label):
    tmp1 = x_label[:, col_tcr]
    tmp2 = pseudo_tcrs


    results = []
    for pid in range(args.num_patients):
        bad_pid = False

        pcts1 = []
        for i in range(2):
            tmp = tmp1[np.logical_and(x_label[:, col_patient] == pid, x_label[:, col_bloodtumor] == i)]
            a, b = np.unique(tmp, return_counts=True)
            if tmp.shape[0] == 0:
                bad_pid = True
                continue
            pcts1.append((b == 1).sum() / tmp.shape[0])
        if bad_pid: continue
        print("Patient {:>2.0f} (real): {:.3f}  {:.3f}".format(pid, pcts1[0], pcts1[1]))

        pcts2 = []
        for i in range(2):
            tmp = tmp2[np.logical_and(x_label[:, col_patient] == pid, x_label[:, col_bloodtumor] == i)]
            a, b = np.unique(tmp, return_counts=True)
            pcts2.append((b == 1).sum() / tmp.shape[0])
        print("Patient {:>2.0f} (pred): {:.3f}  {:.3f}".format(pid, pcts2[0], pcts2[1]))
        print()

        results.append(pcts1 + pcts2)

    results = np.array(results)
    print("median per blood/tumor (real): {}".format(["{:.3f}".format(tmp) for tmp in np.median(results[:, :2], axis=0)]))
    print("median per blood/tumor (pred): {}".format(["{:.3f}".format(tmp) for tmp in np.median(results[:, 2:], axis=0)]))

def get_model_predictions(args, G, load_eval, learn_emb_batch_x_rna, learn_emb_batch_tcr):
    ######################################################################
    print('Getting predictions...')
    t = time.time()

    G.eval()

    preds_rna = []
    preds_tcr = []
    recon_rna = []
    recon_tcr = []
    recon_rna_z = []
    recon_tcr_z = []


    #######################
    # calculate embeddings and fix them
    batch_embeddings_rna = G.mlp_patient_embeddings_rna(learn_emb_batch_x_rna)
    batch_embeddings_tcr = G.mlp_patient_embeddings_tcr(learn_emb_batch_tcr)
    batch_embeddings = torch.stack([batch_embeddings_rna, batch_embeddings_tcr]).mean(axis=0)

    G.bloodtumor_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_bloodtumor], num_labels=2)[0]
    G.prepost_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_prepost], num_labels=2)[0]
    G.patient_embeddings_matrix = G.groupby_mean(batch_embeddings[learn_emb_mask_bb], labels=learn_emb_batch_labels[learn_emb_mask_bb, col_patient], num_labels=args.num_patients)[0]
    #######################

    for batch_x_rna, batch_x_tcr, batch_labels in load_eval.iter_batches(batch_size=args.batch_size):
        batch_random = np.random.normal(0, 1, [batch_x_rna.shape[0], args.dimz])
        
        batch_x_rna = numpy2torch(batch_x_rna)
        batch_x_tcr = numpy2torch(batch_x_tcr)
        batch_labels = numpy2torch(batch_labels, type=torch.IntTensor)
        batch_random = numpy2torch(batch_random)
        

        batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, batch_labels[:, col_bloodtumor].type(torch.int32))
        batch_prepost_embeddings  = torch.index_select(G.prepost_embeddings_matrix, 0, batch_labels[:, col_prepost].type(torch.int32))
        batch_patient_embeddings = torch.index_select(G.patient_embeddings_matrix, 0, batch_labels[:, col_patient].type(torch.int32))


        out_rna_recon, out_tcr_recon, [_, _, [z_rna_recon, z_tcr_recon]] = G(x=[batch_x_rna, batch_x_tcr],
                                               embeddings=[batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings])

        out_rna, out_tcr, [_, _, _] = G.sample(z=batch_random,
                                               embeddings=[batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings])

        preds_rna.append(out_rna.detach().cpu().numpy())
        preds_tcr.append(out_tcr.detach().cpu().numpy())
        recon_rna.append(out_rna_recon.detach().cpu().numpy())
        recon_tcr.append(out_tcr_recon.detach().cpu().numpy())
        recon_rna_z.append(z_rna_recon.detach().cpu().numpy())
        recon_tcr_z.append(z_tcr_recon.detach().cpu().numpy())

    preds_rna = np.concatenate(preds_rna, axis=0)
    preds_tcr = np.concatenate(preds_tcr, axis=0)
    recon_rna = np.concatenate(recon_rna, axis=0)
    recon_tcr = np.concatenate(recon_tcr, axis=0)
    recon_rna_z = np.concatenate(recon_rna_z, axis=0)
    recon_tcr_z = np.concatenate(recon_tcr_z, axis=0)

    print('Got predictions in {:.1f} s'.format(time.time() - t))
    ######################################################################

    return G, preds_rna, preds_tcr, recon_rna, recon_tcr, recon_rna_z, recon_tcr_z

def plot_rna_umap(args, preds_rna, viz_reducer, e_eval_reals, mask_preprocess, x_label):
    ######################################################################
    print('Starting RNA plotting...')
    e_eval_preds = viz_reducer.transform(preds_rna)

    for i_pid in range(args.num_patients):
        print('Patient {}'.format(i_pid))

        mask1 = np.logical_and(np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0), x_label[:, col_patient] == i_pid)
        mask2 = np.logical_and(np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 1), x_label[:, col_patient] == i_pid)
        mask3 = np.logical_and(np.logical_and(x_label[:, col_bloodtumor] == 1, x_label[:, col_prepost] == 0), x_label[:, col_patient] == i_pid)
        mask4 = np.logical_and(np.logical_and(x_label[:, col_bloodtumor] == 1, x_label[:, col_prepost] == 1), x_label[:, col_patient] == i_pid)


        fig.clf()
        fig.suptitle('Patient {}'.format(i_pid))
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        [make_legend(ax, ['Real', 'Predicted'], cmap=mpl.cm.viridis, fontsize=6) for ax in axes.flatten()]
        scatter_helper(e_eval_reals[mask_preprocess][mask1], e_eval_preds[mask1], ax=axes[0, 0])
        scatter_helper(e_eval_reals[mask_preprocess][mask2], e_eval_preds[mask2], ax=axes[0, 1])
        scatter_helper(e_eval_reals[mask_preprocess][mask3], e_eval_preds[mask3], ax=axes[1, 0])
        scatter_helper(e_eval_reals[mask_preprocess][mask4], e_eval_preds[mask4], ax=axes[1, 1])

        axes[0, 0].set_title('Blood Before')
        axes[0, 1].set_title('Blood After')
        axes[1, 0].set_title('Tumor Before')
        axes[1, 1].set_title('Tumor After')
        [[ax.set_xticks([]), ax.set_yticks([])] for ax in axes.flatten()]
        [ax.set_xlabel('UMAP1') for ax in axes[-1, :].flatten()]
        [ax.set_ylabel('UMAP2') for ax in axes[:, 0].flatten()]

        # if i_pid < 4:
        #     fig.savefig('tmp_patient{}.png'.format(i_pid))
        fig.savefig(os.path.join(args.output_folder, 'umap_rna_patient{}.png'.format(i_pid)))
    print('Done with RNA plotting.')
    ######################################################################



x_rna = data_rna
x_tcr = data_tcr
x_label = data_labels

mask_preprocess = x_label[:, col_tcr] != ID_NULL_TCR
x_rna = x_rna[mask_preprocess]
x_tcr = x_tcr[mask_preprocess]
x_label = x_label[mask_preprocess]


args.dimrna = x_rna.shape[-1]
args.dimtcr = x_tcr.shape[-1]
args.num_patients = len(np.unique(data_labels[:, col_patient]))



G = Generator(args)
G = G.to(device)
var_list = [G]


param_list = []
for v in var_list:
    param_list.extend(list(v.parameters()))
opt_G = torch.optim.Adam(param_list, lr=args.lr)



mask_train = np.logical_or(x_label[:, col_patient] != args.heldout_patient, np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0))

load_train = Loader([x_rna[mask_train], x_tcr[mask_train], x_label[mask_train]], shuffle=True)
load_eval = Loader([x_rna, x_tcr, x_label], shuffle=False)



# for learning embeddings
learn_emb_batch_x_rna, learn_emb_batch_tcr, learn_emb_batch_labels = load_train.next_batch(args.n_learn_embedding_sample)

learn_emb_batch_x_rna = numpy2torch(learn_emb_batch_x_rna)
learn_emb_batch_tcr = numpy2torch(learn_emb_batch_tcr)
learn_emb_batch_labels = numpy2torch(learn_emb_batch_labels)

learn_emb_mask_bb = torch.logical_and(learn_emb_batch_labels[:, col_bloodtumor] == 0, learn_emb_batch_labels[:, col_prepost] == 0)


losses = []
t = time.time()
while args.i_iter <= args.training_steps:
    args.i_iter += 1
    [v.train() for v in var_list]
    opt_G.zero_grad()
    batch_loss = []
    
    ########################################################################################################################
    # get data from loader
    batch_x_rna, batch_x_tcr, batch_labels = load_train.next_batch(args.batch_size)

    # convert np to torch objects
    batch_x_rna = numpy2torch(batch_x_rna)
    batch_x_tcr = numpy2torch(batch_x_tcr)
    batch_labels = numpy2torch(batch_labels, type=torch.IntTensor)
    
    
    ########################################################################################################################
    # for learning embeddings
    batch_embeddings_rna = G.mlp_patient_embeddings_rna(learn_emb_batch_x_rna)
    batch_embeddings_tcr = G.mlp_patient_embeddings_tcr(learn_emb_batch_tcr)
    batch_embeddings = torch.stack([batch_embeddings_rna, batch_embeddings_tcr]).mean(axis=0)

    G.bloodtumor_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_bloodtumor], num_labels=2)[0]
    G.prepost_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_prepost], num_labels=2)[0]
    G.patient_embeddings_matrix = G.groupby_mean(batch_embeddings[learn_emb_mask_bb], labels=learn_emb_batch_labels[learn_emb_mask_bb, col_patient], num_labels=args.num_patients)[0]

    batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, batch_labels[:, col_bloodtumor].type(torch.int32))
    batch_prepost_embeddings  = torch.index_select(G.prepost_embeddings_matrix, 0, batch_labels[:, col_prepost].type(torch.int32))
    batch_patient_embeddings = torch.index_select(G.patient_embeddings_matrix, 0, batch_labels[:, col_patient].type(torch.int32))


    #################################################
    # model feedforward
    recon_rna, recon_tcr, [mu, logvar, _] = G(x=[batch_x_rna, batch_x_tcr],
                                              embeddings=[batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings])

    
    kl = - (1 + logvar - logvar.exp() - mu.pow(2)).mean()
    batch_loss.append(args.lambda_kl * kl)
    batch_loss.append(args.lambda_recon_rna * ((batch_x_rna - recon_rna)**2).mean())


    real_same_clones = batch_labels[:, col_tcr][np.newaxis, :] == batch_labels[:, col_tcr][:, np.newaxis]
    real_diff_clones = ~real_same_clones
    real_same_clones = torch.logical_and(real_same_clones, torch.eye(args.batch_size).to(device) == 0)

    same_clones_pred = torch.cdist(recon_tcr, recon_tcr, p=1)

    batch_loss.append(args.lambda_recon_tcr * same_clones_pred[real_same_clones].mean())
    batch_loss.append(args.lambda_recon_tcr * F.relu(args.delta_contrastive - same_clones_pred[real_diff_clones]).mean())

    #################################################
    # finish up loop
    
    # embedding regs
    batch_loss.append(args.lambda_embedding_norm * torch.cat([G.bloodtumor_embeddings_matrix**2, G.prepost_embeddings_matrix**2, G.patient_embeddings_matrix**2]).mean())

    # loss stuff
    batch_loss_list = batch_loss
    batch_loss = torch.mean(torch.stack(batch_loss))
    losses.append(batch_loss.item())

    batch_loss.backward()
    check_for_nan(G)
    opt_G.step()
    opt_G.zero_grad()

    if args.i_iter % args.print_every == 0:
        print("{:>5}: avg loss: {:.3f} ({:.1f} s)".format(args.i_iter, np.mean(losses), time.time() - t ))
        if args.i_iter % (10 * args.print_every) == 0:
            print("{:>5}: batchloss list: {}".format(args.i_iter, ['{:.3f}'.format(l.detach().cpu().numpy()) for l in batch_loss_list]))
        t = time.time()
        losses = []

    if args.i_iter % args.save_every == 0:
        print('Saving args...')
        with open(os.path.join(args.output_folder, 'args.txt'), 'w+') as f:
            json.dump(args.__dict__, f, indent=2)
        print('Args saved!')


        print('Saving model...')
        torch.save(G.state_dict(), os.path.join(args.output_folder, 'model.pth'))
        print('Model saved!')



####################################
####################################
##### post-process, plot results

print('Done training, starting post-analysis...')
t = time.time()

G, preds_rna, preds_tcr, recon_rna, recon_tcr, recon_rna_z, recon_tcr_z = get_model_predictions(args, G, load_eval, learn_emb_batch_x_rna, learn_emb_batch_tcr)

plot_rna_umap(args, preds_rna, viz_reducer, e_eval_reals, mask_preprocess, x_label)

tcr_dists = sklearn.metrics.pairwise_distances(preds_tcr, preds_tcr, metric='l1')

pseudo_tcrs, thresh_fitted = get_pseudoclones(preds_tcr, x_label, mask_train, tcr_dists, thresh=args.delta_contrastive)
check_pseudoclones_per_condition(pseudo_tcrs, x_label)

print('Done with post-analysis in {:.1f} s!'.format(time.time() - t))

####################################
####################################


####################################
####################################
##### saving out

print('Saving args...')
with open(os.path.join(args.output_folder, 'args.txt'), 'w+') as f:
    json.dump(args.__dict__, f)
print('Args saved!')


print('Saving model...')
torch.save(G.state_dict(), os.path.join(args.output_folder, 'model.pth'))
print('Model saved!')


print('Saving data...')
with open(os.path.join(args.output_folder, 'preds.npz'), 'wb+') as f:
    np.savez(f, preds_rna=preds_rna, preds_tcr=preds_tcr,
                recon_rna_z=recon_rna_z, recon_tcr_z=recon_tcr_z,
                pseudo_tcrs=pseudo_tcrs, tcr_dists=tcr_dists, thresh_fitted=thresh_fitted,
                learn_emb_batch_x_rna=learn_emb_batch_x_rna.cpu().numpy(), 
                learn_emb_batch_tcr=learn_emb_batch_tcr.cpu().numpy(),
                learn_emb_batch_labels=learn_emb_batch_labels.cpu().numpy())
print('Data saved!')

####################################
####################################
















