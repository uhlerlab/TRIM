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
    parser.add_argument('--heldout_patient', nargs='+', type=int, default=[3, 4, 13], help='List of patient IDs to leave out (e.g. --leave_out_patient 3 4 13)')

    # data location
    parser.add_argument('--data_parent_folder', default='/data/che/TRIM/CRC')
    parser.add_argument('--tcr_ae_train_step', type=int, default=50100)
    parser.add_argument('--umap_trained_file', default='umap_trained_rna.pkl')
    parser.add_argument('--pca_file', default='data_rna_pca.pkl')

    # training
    parser.add_argument('--training_steps', default=20000)
    parser.add_argument('--lr', default=.001)
    parser.add_argument('--print_every', default=100)
    parser.add_argument('--save_every', default=10000)
    parser.add_argument('--device', default='cuda:1')

    # model architecture
    parser.add_argument('--learn_patient_embeddings', default=1)
    parser.add_argument('--reduction', default='PCA', choices=['PCA', 'HVG'])
    parser.add_argument('--delta_contrastive', default=10)

    # model size
    parser.add_argument('--n_learn_embedding_sample', default=5120, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
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
args.processed_data_folder = args.data_parent_folder
print('Processed data folder: {}'.format(args.processed_data_folder))

args.tcr_folder = os.path.join(args.processed_data_folder, f'tcr_ae/step_{args.tcr_ae_train_step}')
print('TCR folder: {}'.format(args.tcr_folder))

args.umap_trained_file = os.path.join(args.processed_data_folder, args.umap_trained_file)
print('UMAP file path: {}'.format(args.umap_trained_file))

args.pca_file = os.path.join(args.processed_data_folder, args.pca_file)
print('PCA file path: {}'.format(args.pca_file))

if len(args.heldout_patient) == 1:
    output_folder_1 = 'output'
elif args.heldout_patient == [3, 4, 13]:
    output_folder_1 = 'leave_group_out_gender'
elif args.heldout_patient == [12, 13, 18, 21]:
    output_folder_1 = 'leave_group_out_young'
elif args.heldout_patient == [2, 6, 7, 11, 20]:
    output_folder_1 = 'leave_group_out_old'
elif args.heldout_patient == [1, 12, 19, 20]:
    output_folder_1 = 'leave_group_out_few_cells'
elif args.heldout_patient == [0, 3, 5, 10]:
    output_folder_1 = 'leave_group_out_many_cells'

args.output_folder = os.path.join(args.tcr_folder, f'{output_folder_1}/holdout{"_".join(map(str, args.heldout_patient))}')
if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)
print('Output folder: {}'.format(args.output_folder))

print('Batch size: {}'.format(args.batch_size))

###############
############### START data loading
############### 
print('Starting to load data...')
t = time.time()

with open(os.path.join(args.processed_data_folder, 'data_rna.pkl'), 'rb') as f:
    data_rna = pickle.load(f)
print(f'loaded data_rna from {args.processed_data_folder}')
print('data_rna shape: {}'.format(data_rna.shape))

with open(os.path.join(args.tcr_folder, 'data_tcr.pkl'), 'rb') as f:
    data_tcr = pickle.load(f)
print(f'loaded data_tcr from {args.tcr_folder}')
print('data_tcr shape: {}'.format(data_tcr.shape))

with open(os.path.join(args.processed_data_folder, 'data_labels.pkl'), 'rb') as f:
    data_labels = pickle.load(f)
print(f'loaded data_labels from {args.processed_data_folder}')

with open(os.path.join(args.processed_data_folder, 'data_labels_str.pkl'), 'rb') as f:
    data_labels_str = pickle.load(f)
print(f'loaded data_labels_str from {args.processed_data_folder}')

with open(os.path.join(args.processed_data_folder, 'df_all_tcrs.pkl'), 'rb') as f:
    df_all_tcrs = pickle.load(f)
print(f'loaded df_all_tcrs from {args.processed_data_folder}')

# Get column indexes
col_bloodtumor = data_labels.columns.get_loc('Tissue')
col_prepost = data_labels.columns.get_loc('Treatment Stage')
col_celltype = data_labels.columns.get_loc('SubCellType')
col_patient = data_labels.columns.get_loc('Patient')
col_tcr = data_labels.columns.get_loc('CDR3(Beta1)')

ID_NULL_TCR = -1
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
    if not os.path.exists(args.pca_file):
        pca = sklearn.decomposition.PCA(npca, random_state=0)
        data_rna = pca.fit_transform(data_rna)
        pickle.dump(data_rna, open(args.pca_file, 'wb+'))
        print('RNA PCA saved to {}'.format(args.pca_file))
    else:
        data_rna = pickle.load(open(args.pca_file, 'rb'))
        print('RNA PCA loaded from {}'.format(args.pca_file))
    assert data_rna.shape[1] == npca, "RNA PCA shape incorrect"
    assert data_rna.shape[0] == data_labels.shape[0], "RNA PCA num rows incorrect"
elif args.reduction == 'HVG':
    nhvg = 1000
    mask_hvg = scanpy.pp.highly_variable_genes(scanpy.AnnData(data_rna, data_rna.index.to_frame(), data_rna.columns.to_frame()), inplace=False, n_top_genes=nhvg)['highly_variable'].values

print('After {} reduction, data_rna shape: {}'.format(args.reduction, data_rna.shape))

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
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


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


def numpy2torch(x, type=torch.FloatTensor):

    return torch.from_numpy(x).type(type).to(device)

def one_hot_along_3rd_axis(x):
    out = np.zeros([x.shape[0], x.shape[1], len(vocab)])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j, x[i, j]] = 1
    return out

def reparameterize(mu, logvar, clamp=5):
    std = logvar.mul(0.5).exp_()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)

    # std = torch.exp(0.5 * logvar) # standard deviation
    # eps = torch.randn_like(std) # `randn_like` as we need the same size
    # sample = mu + (eps * std) # sampling as if coming from the input space
    # return sample


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
x_label = data_labels.values

mask_preprocess = x_label[:, col_tcr] != ID_NULL_TCR
assert mask_preprocess.sum() == x_rna.shape[0], "Some cells with RNA don't have TCR"
x_rna = x_rna[mask_preprocess]
x_tcr = x_tcr[mask_preprocess]
x_label = x_label[mask_preprocess]


args.dimrna = x_rna.shape[-1]
args.dimtcr = x_tcr.shape[-1]
args.num_patients = len(np.unique(data_labels.values[:, col_patient]))
print('RNA dim: {}, TCR dim: {}, Num patients: {}'.format(args.dimrna, args.dimtcr, args.num_patients))
assert x_label[:, col_patient].max() == args.num_patients - 1, "Patient IDs should be 0-indexed"
assert (np.isin(x_label[:, col_patient], args.heldout_patient)).sum() > 0, "Heldout patient not in data"

G = Generator(args)
G = G.to(device)
var_list = [G]


param_list = []
for v in var_list:
    param_list.extend(list(v.parameters()))
opt_G = torch.optim.Adam(param_list, lr=args.lr)



mask_train = np.logical_or(~np.isin(x_label[:, col_patient], args.heldout_patient), np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0))

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
mask_leave_out = (np.isin(x_label[:, col_patient], args.heldout_patient))
assert np.unique(x_label[mask_leave_out][:, col_patient]).tolist() == args.heldout_patient, "mask_leave_out incorrect"
if len(args.heldout_patient) == 1:
    with open(os.path.join(args.output_folder, 'preds.npz'), 'wb+') as f:
        np.savez(f, preds_rna=preds_rna[mask_leave_out], preds_tcr=preds_tcr[mask_leave_out],
                    recon_rna_z=recon_rna_z[mask_leave_out], recon_tcr_z=recon_tcr_z[mask_leave_out],
                    pseudo_tcrs=pseudo_tcrs[mask_leave_out], thresh_fitted=thresh_fitted,
                    # tcr_dists=tcr_dists, 
                    learn_emb_batch_x_rna=learn_emb_batch_x_rna.cpu().numpy(), 
                    learn_emb_batch_tcr=learn_emb_batch_tcr.cpu().numpy(),
                    learn_emb_batch_labels=learn_emb_batch_labels.cpu().numpy())
else:
    # save to individual folders for each heldout patient for clarity
    for pid in args.heldout_patient:
        mask_leave_out = x_label[:, col_patient] == pid
        # get the parent of args.output_folder
        parent_path = os.path.dirname(args.output_folder)
        individual_folder = os.path.join(parent_path, f'holdout{pid}')
        if not os.path.exists(individual_folder):
            os.makedirs(individual_folder)
            print(f'Created folder: {individual_folder}')
        with open(os.path.join(individual_folder, f'preds.npz'), 'wb+') as f:
            np.savez(f, preds_rna=preds_rna[mask_leave_out], preds_tcr=preds_tcr[mask_leave_out],
                        recon_rna_z=recon_rna_z[mask_leave_out], recon_tcr_z=recon_tcr_z[mask_leave_out],
                        pseudo_tcrs=pseudo_tcrs[mask_leave_out], thresh_fitted=thresh_fitted,
                        # tcr_dists=tcr_dists, 
                        learn_emb_batch_x_rna=learn_emb_batch_x_rna.cpu().numpy(), 
                        learn_emb_batch_tcr=learn_emb_batch_tcr.cpu().numpy(),
                        learn_emb_batch_labels=learn_emb_batch_labels.cpu().numpy())

        args.heldout_patient = [pid]  # update args for each individual
        with open(os.path.join(individual_folder, 'args.txt'), 'w+') as f:
            json.dump(args.__dict__, f)

        torch.save(G.state_dict(), os.path.join(individual_folder, 'model.pth'))
        print(f'Saved data for heldout patient {pid} in {individual_folder}')

print('Data saved!')

####################################
####################################

print('All done!')
















