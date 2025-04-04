import os
import numpy as np
import random
random.seed(0)
np.random.seed(0)

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

### Helper functions
#############################################
#############################################

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

def reparameterize(mu, logvar, clamp=5):
    std = logvar.mul(0.5).exp_()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)

def numpy2torch(x, type=torch.FloatTensor):

    return torch.from_numpy(x).type(type).to(device)

def make_legend(ax, labels, s=20, cmap=mpl.cm.jet, center=None, **kwargs):
    if center is None:
        center = [0, 0]
    uniquelabs = np.unique(labels)
    numlabs = len(uniquelabs)
    for i, label in enumerate(uniquelabs):
        if numlabs > 1:
            ax.scatter(center[0], center[1], s=s, c=[cmap(1 * i / (numlabs - 1))], label=label)
        else:
            ax.scatter(center[0], center[1], s=s, c=[cmap(1.)], label=label)
    ax.scatter(center[0], center[1], s=2 * s, c='w')
    ax.legend(**kwargs)

def make_legend2(ax, labels, s=20, cmap=mpl.cm.jet, color_order=None, center=None, **kwargs):
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

def truncate_colormap(cmap, vmin=0.0, vmax=1.0, n=100):
    str_ = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax)
    cmap_ = cmap(np.linspace(vmin, vmax, n))
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(str_, cmap_)
    return new_cmap



### Helper functions for model evaluations
#############################################
#############################################

def baseline_expansion_prediction(train_on, baseline_model, num_patients, figname='tmp.png', figformat='png', seed=0):
    # train_on         can be  ['rna', 'tcr', 'both']
    # baseline_model   can be  ['svm', 'knn', 'random_forest', 'NN']
    import sklearn.svm
    import sklearn.neighbors
    import sklearn.ensemble
    import sklearn.neural_network

    np.random.seed(seed)
    torch.manual_seed(seed)

    all_baseline_real = []
    all_baseline_pred = []

    for i_pid in range(num_patients):
        print('holding out patient {}'.format(i_pid))
        mask_train = np.logical_and(x_label[:, col_patient] != i_pid, x_label[:, col_bloodtumor] == 0)
        x_rna_masked_train = x_rna[mask_train]
        x_tcr_masked_train = x_tcr[mask_train]
        labels_masked_train = x_label[mask_train]
        
        blood_expanded_train = (df_all_tcrs.iloc[:, 1].fillna(0) > df_all_tcrs.iloc[:, 0].fillna(0)).values
        blood_expanded_train = np.take(blood_expanded_train, labels_masked_train[:, col_tcr].astype(np.int32))



        mask_test = np.logical_and(x_label[:, col_patient] == i_pid, np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0))
        
        mask_test_new = []
        for i in range(mask_test.shape[0]):
            if not mask_test[i]:
                mask_test_new.append(False)
                continue
            if (x_label[i, col_tcr] == x_label[mask_train, col_tcr]).any():
                mask_test_new.append(False)
                continue
            mask_test_new.append(True)
        mask_test = np.array(mask_test_new)

        x_rna_masked_test = x_rna[mask_test]
        x_tcr_masked_test = x_tcr[mask_test]
        labels_masked_test = x_label[mask_test]
        


        blood_expanded_test = (df_all_tcrs.iloc[:, 1].fillna(0) > df_all_tcrs.iloc[:, 0].fillna(0)).values
        blood_expanded_test = np.take(blood_expanded_test, labels_masked_test[:, col_tcr].astype(np.int32))
        if blood_expanded_test.sum() == 0: continue

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
            classifier = sklearn.svm.SVC(probability=True)
            r_subsample = np.random.choice(range(mask_train.sum()), 10000, replace=False)
        elif baseline_model == 'knn':
            classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=20)
            r_subsample = np.random.choice(range(mask_train.sum()), mask_train.sum(), replace=False)
        elif baseline_model == 'random_forest':
            classifier = sklearn.ensemble.RandomForestClassifier(random_state=0)
            r_subsample = np.random.choice(range(mask_train.sum()), mask_train.sum(), replace=False)
        elif baseline_model == 'NN':
            classifier = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=[400, 200, 100])
            r_subsample = np.random.choice(range(mask_train.sum()), mask_train.sum(), replace=False)
        else:
            raise Exception("bad baseline_model")


        mask_drop_dup_train = get_drop_duplicates_mask(classifier_x_train[r_subsample])
        mask_drop_dup_test = get_drop_duplicates_mask(classifier_x_test)

        classifier.fit(classifier_x_train[r_subsample][mask_drop_dup_train], blood_expanded_train[r_subsample][mask_drop_dup_train])
        preds = classifier.predict_proba(classifier_x_test[mask_drop_dup_test])[:, 1]

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(blood_expanded_test[mask_drop_dup_test], preds, pos_label=1)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print("ROC: {:.3f}".format(roc_auc))

        all_baseline_real.append(blood_expanded_test[mask_drop_dup_test])
        all_baseline_pred.append(preds)

    all_baseline_real = np.concatenate(all_baseline_real, axis=0)
    all_baseline_pred = np.concatenate(all_baseline_pred, axis=0)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(all_baseline_real, all_baseline_pred, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print(roc_auc)


    fig.clf()
    ax = fig.subplots(1, 1)
    lw = 2
    ax.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    [ax.set_xlim([0.0, 1.0]), ax.set_ylim([0.0, 1.05])]
    [ax.set_xlabel("False Positive Rate"), ax.set_ylabel("True Positive Rate")]
    ax.legend(loc="lower right")
    fig.savefig(figname, format=figformat)
    fig.savefig(figname.replace('.png', '.svg'), format='svg')


def our_expansion_prediction(output_folders, num_points=10000):
    results_across_experiments_real = []
    results_across_experiments_pseudo = []

    for output_folder in tqdm(output_folders):
        print(output_folder)

        #############################################
        # load args

        # output_folder_full = os.path.join(os.path.expanduser('~'), 'tcr', 'output', output_folder)
        output_folder_full = os.path.join('/data/che/TRIM/HNSCC/output', output_folder)
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

            pseudo_tcrs = npzfile['pseudo_tcrs']
            tcr_dists = npzfile['tcr_dists']
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
        for condition in range(4):

            dists = sklearn.metrics.pairwise_distances(all_out_tcr[condition], all_out_tcr[condition], metric='l1')
            all_dists.append(dists)
            print('dists {} done'.format(condition))


        all_pseudo_tcrs = []
        for condition in range(4):

            dists = all_dists[condition]

            thresh = thresh_fitted / 2
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
            for condition in range(4):
                pseudo_id = all_pseudo_tcrs[condition, i]
                clonality = (all_pseudo_tcrs[condition] == pseudo_id).sum()
                clonalities.append(clonality)
            all_pseudo_clonalities.append(clonalities)

        all_pseudo_clonalities = np.array(all_pseudo_clonalities)
        all_pseudo_clonalities = np.mean(np.array(np.array_split(all_pseudo_clonalities, num_samples)), 0)


        all_real_clonalities = np.take(df_all_tcrs.fillna(0).values, x_label[mask_pid_bb, col_tcr].astype(int), axis=0)


        results_across_experiments_real.append(all_real_clonalities)
        results_across_experiments_pseudo.append(all_pseudo_clonalities)

    return results_across_experiments_real, results_across_experiments_pseudo


