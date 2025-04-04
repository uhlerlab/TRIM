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
old_fig_size = fig.get_size_inches()



def get_args():
    parser = argparse.ArgumentParser()

    # model ID
    parser.add_argument('--output_folder', default='/home/che/tcr/output/test')
    parser.add_argument('--heldout_patient', default=0)

    # data location
    parser.add_argument('--processed_data_folder', default='/home/che/tcr/data_processed')
    parser.add_argument('--umap_trained_file', default='/home/che/tcr/umap_trained.pkl')

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
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--n_channels_base', default=256, type=int)
    parser.add_argument('--dimz', default=1024)
    parser.add_argument('--dim_state_embedding', default=128)
    
    # loss terms
    parser.add_argument('--lambda_kl', default=.1)
    parser.add_argument('--lambda_recon_rna', default=1)
    parser.add_argument('--lambda_recon_tcr', default=1)
    parser.add_argument('--lambda_embedding_norm', default=10)

    args = parser.parse_args()

    args.i_iter = 0

    return args

args = get_args()

if not os.path.exists(args.output_folder): 
    os.mkdir(args.output_folder)


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




#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def numpy2torch(x, type=torch.FloatTensor):

    return torch.from_numpy(x).type(type).to(device)

def one_hot_along_3rd_axis(x):
    out = np.zeros([x.shape[0], x.shape[1], len(vocab)])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j, x[i, j]] = 1
    return out

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

class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        nbase = self.args.n_channels_base

        n_conditions = 3 * args.dim_state_embedding

        # RNA
        total_dim_in_rna = args.dimrna + n_conditions
        self.encoder_rna = MLP(dim_in=total_dim_in_rna, dim_out=args.dimz * 2, nbase=nbase * 1)

        # TCR
        total_dim_in_tcr = args.dimtcr + n_conditions
        self.encoder_tcr = MLP(dim_in=total_dim_in_tcr, dim_out=args.dimz * 2, nbase=nbase * 1)

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

        # shared_layer = torch.cat([z] + embeddings, axis=-1)

        # recon_tcr = self.decoder_tcr(shared_layer)
        # recon_rna = self.decoder_rna(shared_layer)

        return z, mu, logvar

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

        out = F.softmax(out, dim=-1)

        return out

def get_optimal_cutoff(fpr, tpr, thresholds):
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    roc = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return roc['thresholds'].values[0]

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


args.training_steps = 2000
all_out_grads = []
#####

for _ in range(10):
    x_rna = data_rna
    x_tcr = data_tcr
    x_label = data_labels

    mask_preprocess = x_label[:, col_tcr] != ID_NULL_TCR
    x_rna = x_rna[mask_preprocess]
    x_tcr = x_tcr[mask_preprocess]
    x_label = x_label[mask_preprocess]




    blood_expanded = np.logical_and(df_all_tcrs.fillna(0).iloc[:, 0] > 0, df_all_tcrs.fillna(0).iloc[:, 1] > 0) #have counts blood before and after
    blood_expanded = np.logical_and(blood_expanded, df_all_tcrs.fillna(0).iloc[:, 0] < df_all_tcrs.fillna(0).iloc[:, 1]) #CH: have more counts after
    blood_expanded = np.take(blood_expanded.values, x_label[:, col_tcr].astype(np.int32), axis=0)

    # blood_expanded = (df_all_tcrs.fillna(0).iloc[:, 1] > df_all_tcrs.fillna(0).iloc[:, 0]).astype(np.int32).values
    # blood_expanded = np.take(blood_expanded, x_label[:, col_tcr].astype(np.int32))

    mask_blood = np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0)
    mask_blood = np.logical_and(mask_blood, x_label[:, col_celltype] == 1)
    x_rna = x_rna[mask_blood]
    x_tcr = x_tcr[mask_blood]
    x_label = x_label[mask_blood]
    blood_expanded = blood_expanded[mask_blood]

    args.dimrna = x_rna.shape[-1]
    args.dimtcr = x_tcr.shape[-1]
    args.num_patients = len(np.unique(data_labels[:, col_patient]))
    args.i_iter = 0

    G = Generator(args)
    G = G.to(device)
    decoder_rna = MLP(dim_in=args.dimz + 3 * args.dim_state_embedding, dim_out=args.dimrna, nbase=args.n_channels_base * 1).to(device)
    classifier = MLP(nbase=args.n_channels_base, dim_in=args.dimz*1, dim_out=2)
    classifier = classifier.to(device)
    var_list = [G, classifier]

    ce = nn.CrossEntropyLoss()

    param_list = []
    for v in var_list:
        param_list.extend(list(v.parameters()))
    opt_G = torch.optim.Adam(param_list, lr=args.lr)#, weight_decay=.001)


    args.heldout_patient = 1
    # mask_train = np.logical_or(x_label[:, col_patient] != args.heldout_patient, np.logical_and(x_label[:, col_bloodtumor] == 0, x_label[:, col_prepost] == 0))
    mask_train = np.logical_and(x_label[:, col_patient] != args.heldout_patient, x_label[:, col_prepost] == 0)
    load_train = Loader([x_rna[mask_train], x_tcr[mask_train], x_label[mask_train], blood_expanded[mask_train]], shuffle=True)
    load_eval = Loader([x_rna, x_tcr, x_label, blood_expanded], shuffle=False)




    # for learning embeddings
    learn_emb_batch_x_rna, learn_emb_batch_tcr, learn_emb_batch_labels, _ = load_train.next_batch(args.n_learn_embedding_sample)

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
        batch_x_rna, batch_x_tcr, batch_labels, batch_blood_expanded = load_train.next_batch(args.batch_size)

        # convert np to torch objects
        batch_x_rna = numpy2torch(batch_x_rna)
        batch_x_tcr = numpy2torch(batch_x_tcr)
        batch_labels = numpy2torch(batch_labels, type=torch.IntTensor)
        batch_blood_expanded = numpy2torch(batch_blood_expanded, type=torch.long)
        
        
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
        z, mu, logvar = G(x=[batch_x_rna, batch_x_tcr],
                          embeddings=[batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings])

        pred_z = classifier(mu)

        # recon = decoder_rna(torch.cat([mu, batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings], axis=-1))
        # batch_loss.append(((batch_x_rna - recon)**2).mean())

        class_weights = [batch_blood_expanded.shape[0] / (batch_blood_expanded == 0).sum(), batch_blood_expanded.shape[0] / (batch_blood_expanded == 1).sum()]
        # class_weights = torch.where(batch_blood_expanded == 0, class_weights[0], 3 * class_weights[1])[:, np.newaxis]
        # CH: update here
        class_weights = torch.where(batch_blood_expanded == 0, class_weights[0], class_weights[1])[:, np.newaxis]
        bce_loss = F.binary_cross_entropy(pred_z, F.one_hot(batch_blood_expanded).type(torch.float32), weight=class_weights)
        batch_loss.append(bce_loss.mean())

        # kl = - (1 + logvar - logvar.exp() - mu.pow(2)).mean()
        # batch_loss.append(.1 * args.lambda_kl * kl)

        #################################################
        # finish up loop
        
        # embedding regs
        batch_loss.append(args.lambda_embedding_norm * torch.cat([G.bloodtumor_embeddings_matrix**2, G.prepost_embeddings_matrix**2, G.patient_embeddings_matrix**2]).mean())

        # # l1 reg
        # batch_loss.append(.1 * torch.stack([torch.abs(p).mean() for name, p in G.named_parameters() if 'rna' in name and  'weight' in name]).mean())

        # loss stuff
        batch_loss_list = batch_loss
        batch_loss = torch.mean(torch.stack(batch_loss))
        losses.append(batch_loss.item())

        batch_loss.backward()
        opt_G.step()
        opt_G.zero_grad()

        if args.i_iter % args.print_every == 0:
            print("{:>5}: avg loss: {:.3f} ({:.1f} s)".format(args.i_iter, np.mean(losses), time.time() - t ))
            if args.i_iter % (10 * args.print_every) == 0:
                print("{:>5}: batchloss list: {}".format(args.i_iter, ['{:.3f}'.format(l.detach().cpu().numpy()) for l in batch_loss_list]))
            t = time.time()
            losses = []

        if args.i_iter % 500 == 0:
            [v.eval() for v in var_list]
            batch_embeddings_rna = G.mlp_patient_embeddings_rna(learn_emb_batch_x_rna)
            batch_embeddings_tcr = G.mlp_patient_embeddings_tcr(learn_emb_batch_tcr)
            batch_embeddings = torch.stack([batch_embeddings_rna, batch_embeddings_tcr]).mean(axis=0)

            G.bloodtumor_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_bloodtumor], num_labels=2)[0]
            G.prepost_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_prepost], num_labels=2)[0]
            G.patient_embeddings_matrix = G.groupby_mean(batch_embeddings[learn_emb_mask_bb], labels=learn_emb_batch_labels[learn_emb_mask_bb, col_patient], num_labels=args.num_patients)[0]


            preds = []
            for batch_x_rna, batch_x_tcr, batch_labels, _ in load_eval.iter_batches(args.batch_size):
                batch_x_rna = numpy2torch(batch_x_rna)
                batch_x_tcr = numpy2torch(batch_x_tcr)
                batch_labels = numpy2torch(batch_labels, type=torch.IntTensor)

                batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, batch_labels[:, col_bloodtumor].type(torch.int32))
                batch_prepost_embeddings  = torch.index_select(G.prepost_embeddings_matrix, 0, batch_labels[:, col_prepost].type(torch.int32))
                batch_patient_embeddings = torch.index_select(G.patient_embeddings_matrix, 0, batch_labels[:, col_patient].type(torch.int32))

                z, mu, logvar = G(x=[batch_x_rna, batch_x_tcr],
                                  embeddings=[batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings])

                pred_z = classifier(mu)
                preds.append(pred_z.detach().cpu().numpy())

            preds = np.concatenate(preds)



            print()
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(blood_expanded[mask_train], preds[:, 1][mask_train], pos_label=1)
            print("ROC train: {:.3f}".format(sklearn.metrics.auc(fpr, tpr)))


            fpr, tpr, thresholds = sklearn.metrics.roc_curve(blood_expanded[~mask_train], preds[:, 1][~mask_train], pos_label=1)
            print("ROC  test: {:.3f}".format(sklearn.metrics.auc(fpr, tpr)))

            # cutoff = .004 #get_optimal_cutoff(fpr, tpr, thresholds)
            # CH: update here
            cutoff = get_optimal_cutoff(fpr, tpr, thresholds)
            acc = (preds[:, 1] > cutoff)== blood_expanded
            mask1 = mask_train
            mask2 = blood_expanded == 0

            print('Accuracy on true\n       non-expanded / expanded\n')
            print("Train:    {:.3f}         {:.3f}".format(acc[np.logical_and(mask1, mask2)].mean(), acc[np.logical_and(mask1, ~mask2)].mean()))
            print("Test :    {:.3f}         {:.3f}".format(acc[np.logical_and(~mask1, mask2)].mean(), acc[np.logical_and(~mask1, ~mask2)].mean()))
            print()


    [v.eval() for v in var_list]
    preds = []
    for batch_x_rna, batch_x_tcr, batch_labels, _ in load_eval.iter_batches(args.batch_size):
        batch_x_rna = numpy2torch(batch_x_rna)
        batch_x_tcr = numpy2torch(batch_x_tcr)
        batch_labels = numpy2torch(batch_labels, type=torch.IntTensor)

        batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, batch_labels[:, col_bloodtumor].type(torch.int32))
        batch_prepost_embeddings  = torch.index_select(G.prepost_embeddings_matrix, 0, batch_labels[:, col_prepost].type(torch.int32))
        batch_patient_embeddings = torch.index_select(G.patient_embeddings_matrix, 0, batch_labels[:, col_patient].type(torch.int32))

        _, mu, _ = G(x=[batch_x_rna, batch_x_tcr],
                     embeddings=[batch_bloodtumor_embeddings, batch_prepost_embeddings, batch_patient_embeddings])

        preds_ = classifier(mu)

        preds.append(preds_.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    print(preds.shape)


         

    [v.eval() for v in var_list]
    batch_embeddings_rna = G.mlp_patient_embeddings_rna(learn_emb_batch_x_rna)
    batch_embeddings_tcr = G.mlp_patient_embeddings_tcr(learn_emb_batch_tcr)
    batch_embeddings = torch.stack([batch_embeddings_rna, batch_embeddings_tcr]).mean(axis=0)

    G.bloodtumor_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_bloodtumor], num_labels=2)[0]
    G.prepost_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=learn_emb_batch_labels[:, col_prepost], num_labels=2)[0]
    G.patient_embeddings_matrix = G.groupby_mean(batch_embeddings[learn_emb_mask_bb], labels=learn_emb_batch_labels[learn_emb_mask_bb, col_patient], num_labels=args.num_patients)[0]


    out_grads = []
    for batch_x_rna, batch_x_tcr, batch_labels, _ in load_eval.iter_batches(args.batch_size):
        print('batch {} / {}'.format(len(out_grads), load_eval.data[0].shape[0] // args.batch_size))
        batch_x_rna = numpy2torch(batch_x_rna)
        batch_x_tcr = numpy2torch(batch_x_tcr)
        batch_labels = numpy2torch(batch_labels, type=torch.IntTensor)

        batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, batch_labels[:, col_bloodtumor].type(torch.int32))
        batch_prepost_embeddings  = torch.index_select(G.prepost_embeddings_matrix, 0, batch_labels[:, col_prepost].type(torch.int32))
        batch_patient_embeddings = torch.index_select(G.patient_embeddings_matrix, 0, batch_labels[:, col_patient].type(torch.int32))

        
        # pred_z = classifier(mu)

        out_grad = []
        for i in range(batch_x_rna.shape[0]):
            tmp_rna = batch_x_rna[i][np.newaxis, :]
            tmp_rna.requires_grad = True
            _, mu, _ = G(x=[tmp_rna, batch_x_tcr[i][np.newaxis, :]],
                         embeddings=[batch_bloodtumor_embeddings[i][np.newaxis, :], batch_prepost_embeddings[i][np.newaxis, :], batch_patient_embeddings[i][np.newaxis, :]])

            pred_z = classifier(mu)
            g1 = torch.autograd.grad(outputs=pred_z[:, 1], inputs=tmp_rna, retain_graph=True)[0].cpu().numpy()
            g2 = g1 #-torch.autograd.grad(outputs=pred_z[:, 0], inputs=tmp_rna, retain_graph=True)[0].cpu().numpy()
            
            g = np.mean([g1, g2], axis=0)
            out_grad.append(g)
        out_grad = np.concatenate(out_grad, axis=0)

        out_grads.append(out_grad)
    out_grads = np.concatenate(out_grads, axis=0)
    print(out_grads.shape)


    all_out_grads.append(out_grads)

out_grads = np.array(all_out_grads).mean(axis=0)



with open(os.path.join(args.processed_data_folder, 'salience_grads_new.npz'), 'wb+') as f: 
    np.savez(f, out_grads=out_grads)


with open(os.path.join(args.processed_data_folder, 'combined_data_columns.npz'), 'rb') as f: 
    npzfile = np.load(f, allow_pickle=True) 
    combined_data_columns = npzfile['cols'] 
