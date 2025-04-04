import pandas as pd
import glob
import numpy as np
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
import os
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

def library_size_normalize(df, eps=1e-6):
    # log-scale
    # afterwards --> min/max scaling
    df = np.log(df + eps)
    df = df - df.min(axis=1).values[:, np.newaxis]
    df = df / df.max(axis=1).values[:, np.newaxis]

    # df = df / df.sum(axis=1).values[:, np.newaxis] * 1000

    return df

data_path = '/data/che/TRIM/panc' #/data/che/panc

######################################################################
######################################################################
######################################################################
# multi-modal VAE

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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_folder', default='/data/che/TRIM/panc/model')


    # training
    parser.add_argument('--training_steps', default=20000)
    parser.add_argument('--lr', default=.001)
    parser.add_argument('--print_every', default=100)
    parser.add_argument('--save_every', default=10000)

    # model architecture
    parser.add_argument('--learn_patient_embeddings', default=1)
    parser.add_argument('--delta_contrastive', default=10)

    # model size
    parser.add_argument('--n_learn_embedding_sample', default=4096, type=int)
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

def numpy2torch(x, type=torch.FloatTensor):

    return torch.from_numpy(x).type(type).to(device)

def reparameterize(mu, logvar, clamp=5):
    std = logvar.mul(0.5).exp_()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


npca = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


fn_embeddings = f'{data_path}/trained_tcr_embeddings_ae.npz'
with open(fn_embeddings, 'rb') as f:
    npzfile = np.load(f)
    embeddings = npzfile['embeddings']
embeddings = embeddings[mask_patient[mask_null]]


args = get_args()
if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)

args.dimrna = npca
args.dimtcr = embeddings.shape[-1]
args.num_patients = len(np.unique(df['patient'].unique()))

dict_loc2num = {'P': 0, 'N': 0, 'T': 1,}
dict_patient2num = {p: i for i, p in enumerate(sorted(df['patient'].unique()))}
dict_cancertype2num = {c: i for i, c in enumerate(sorted(df['cancerType'].unique()))}
df['loc'] = df['loc'].apply(lambda tmp: dict_loc2num[tmp])
df['patient'] = df['patient'].apply(lambda tmp: dict_patient2num[tmp])
df['cancerType'] = df['cancerType'].apply(lambda tmp: dict_cancertype2num[tmp])
col_loc = df.columns[npca:].tolist().index('loc')
col_patient = df.columns[npca:].tolist().index('patient')
col_cancertype = df.columns[npca:].tolist().index('cancerType')



args.i_iter = 0
G = Generator(args)
G = G.to(device)
var_list = [G]

param_list = []
for v in var_list:
    param_list.extend(list(v.parameters()))
opt_G = torch.optim.Adam(param_list, lr=args.lr)



# CH: specify holdout patients
patients_holdout = [4, 12, 18] # this is the base model holdouts
# patients_holdout = [10, 11]

# unique_tcrs = df['cdr3'].unique()
# tcrs_train = set(np.random.choice(unique_tcrs, int(.9 * unique_tcrs.shape[0]), replace=False))
# mask_train = df['cdr3'].apply(lambda tmp: tmp in tcrs_train)
mask_train = np.logical_or(df['patient'].apply(lambda tmp: tmp not in patients_holdout), df['loc'] == 0)


load_train = Loader([df[mask_train].values[:, :npca].astype(np.float32), embeddings[mask_train], df[mask_train].values[:, npca:]], shuffle=True)
load_eval = Loader([df.values[:, :npca].astype(np.float32), embeddings, df.values[:, npca:]], shuffle=False)



learn_emb_batch_x_rna, learn_emb_batch_tcr, learn_emb_batch_labels = load_train.next_batch(args.n_learn_embedding_sample)

learn_emb_batch_x_rna = numpy2torch(learn_emb_batch_x_rna)
learn_emb_batch_tcr = numpy2torch(learn_emb_batch_tcr)

learn_emb_mask_bb = learn_emb_batch_labels[:, col_loc] == 0


train_multi_modal_model = True
if train_multi_modal_model:
    losses = []
    t = time.time()
    while args.i_iter <= args.training_steps:
        args.i_iter += 1
        [v.train() for v in var_list]
        opt_G.zero_grad()
        batch_loss = []
        
        ########################################################################################################################
        # get data from loader
        batch_x_rna, batch_x_tcr, batch_x_labels = load_train.next_batch(args.batch_size)

        # convert np to torch objects
        batch_x_rna = numpy2torch(batch_x_rna)
        batch_x_tcr = numpy2torch(batch_x_tcr)
        
        ########################################################################################################################
        # for learning embeddings
        batch_embeddings_rna = G.mlp_patient_embeddings_rna(learn_emb_batch_x_rna)
        batch_embeddings_tcr = G.mlp_patient_embeddings_tcr(learn_emb_batch_tcr)
        batch_embeddings = torch.stack([batch_embeddings_rna, batch_embeddings_tcr]).mean(axis=0)

        G.cancertype_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=numpy2torch(learn_emb_batch_labels[:, col_cancertype].astype(int)), num_labels=3)[0]
        G.bloodtumor_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=numpy2torch(learn_emb_batch_labels[:, col_loc].astype(int)), num_labels=2)[0]
        G.patient_embeddings_matrix = G.groupby_mean(batch_embeddings[learn_emb_mask_bb], labels=numpy2torch(learn_emb_batch_labels[:, col_patient][learn_emb_mask_bb].astype(int)), num_labels=args.num_patients)[0]

        batch_cancertype_embeddings = torch.index_select(G.cancertype_embeddings_matrix, 0, numpy2torch(batch_x_labels[:, col_cancertype].astype(int)).type(torch.int))
        batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, numpy2torch(batch_x_labels[:, col_loc].astype(int)).type(torch.int))
        batch_patient_embeddings = torch.index_select(G.patient_embeddings_matrix, 0, numpy2torch(batch_x_labels[:, col_patient].astype(int)).type(torch.int))

        #################################################
        # model feedforward
        recon_rna, recon_tcr, [mu, logvar, _] = G(x=[batch_x_rna, batch_x_tcr],
                                                embeddings=[batch_cancertype_embeddings, batch_bloodtumor_embeddings, batch_patient_embeddings])

        
        kl = - (1 + logvar - logvar.exp() - mu.pow(2)).mean()
        batch_loss.append(args.lambda_kl * kl)
        batch_loss.append(args.lambda_recon_rna * ((batch_x_rna - recon_rna)**2).mean())


        real_same_clones = batch_x_tcr.sum(axis=1)[np.newaxis, :] == batch_x_tcr.sum(axis=1)[:, np.newaxis]
        real_diff_clones = ~real_same_clones
        real_same_clones = torch.logical_and(real_same_clones, torch.eye(args.batch_size).to(device) == 0)

        same_clones_pred = torch.cdist(recon_tcr, recon_tcr, p=1)

        batch_loss.append(args.lambda_recon_tcr * same_clones_pred[real_same_clones].mean())
        batch_loss.append(args.lambda_recon_tcr * F.relu(args.delta_contrastive - same_clones_pred[real_diff_clones]).mean())

        #################################################
        # finish up loop
        
        # embedding regs
        batch_loss.append(args.lambda_embedding_norm * torch.cat([G.cancertype_embeddings_matrix**2, G.bloodtumor_embeddings_matrix**2, G.patient_embeddings_matrix**2]).mean())

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

        if args.i_iter % args.save_every == 0:
            print('Saving args...')
            with open(os.path.join(args.output_folder, 'args.txt'), 'w+') as f:
                json.dump(args.__dict__, f, indent=2)
            print('Args saved!')


            print('Saving model...')
            torch.save(G.state_dict(), os.path.join(args.output_folder, 'model.pth'))
            print('Model saved!')
else:
    print('Skipping training of multi-modal VAE')
    print('-'*100)


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

    G.cancertype_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=numpy2torch(learn_emb_batch_labels[:, col_cancertype].astype(int)), num_labels=3)[0]
    G.bloodtumor_embeddings_matrix = G.groupby_mean(batch_embeddings, labels=numpy2torch(learn_emb_batch_labels[:, col_loc].astype(int)), num_labels=2)[0]
    G.patient_embeddings_matrix = G.groupby_mean(batch_embeddings[learn_emb_mask_bb], labels=numpy2torch(learn_emb_batch_labels[:, col_patient][learn_emb_mask_bb].astype(int)), num_labels=args.num_patients)[0]

    #######################

    for batch_x_rna, batch_x_tcr, batch_x_labels in load_eval.iter_batches(batch_size=args.batch_size):
        batch_random = np.random.normal(0, 1, [batch_x_rna.shape[0], args.dimz])
        
        batch_x_rna = numpy2torch(batch_x_rna)
        batch_x_tcr = numpy2torch(batch_x_tcr)
        batch_random = numpy2torch(batch_random)
        

        batch_cancertype_embeddings = torch.index_select(G.cancertype_embeddings_matrix, 0, numpy2torch(batch_x_labels[:, col_cancertype].astype(int)).type(torch.int))
        batch_bloodtumor_embeddings = torch.index_select(G.bloodtumor_embeddings_matrix, 0, numpy2torch(batch_x_labels[:, col_loc].astype(int)).type(torch.int))
        batch_patient_embeddings = torch.index_select(G.patient_embeddings_matrix, 0, numpy2torch(batch_x_labels[:, col_patient].astype(int)).type(torch.int))


        out_rna_recon, out_tcr_recon, [_, _, [z_rna_recon, z_tcr_recon]] = G(x=[batch_x_rna, batch_x_tcr],
                                               embeddings=[batch_cancertype_embeddings, batch_bloodtumor_embeddings, batch_patient_embeddings])

        out_rna, out_tcr, [_, _, _] = G.sample(z=batch_random,
                                               embeddings=[batch_cancertype_embeddings, batch_bloodtumor_embeddings, batch_patient_embeddings])

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

def get_pseudoclones(preds_tcr, x_label, train_mask, dists, thresh, tol=.025, max_tries=50, print_every=False):
    t = time.time()

    num_unique = len(np.unique(x_label[train_mask]['cdr3']))

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
            mask = np.logical_and(mask, x_label.iloc[:, col_loc] == x_label.iloc[i, col_loc])
            mask = np.logical_and(mask, x_label.iloc[:, col_patient] == x_label.iloc[i, col_patient])
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


print('Done training, starting post-analysis...')
run_post_analysis = True
if run_post_analysis:
    t = time.time()

    G, preds_rna, preds_tcr, recon_rna, recon_tcr, recon_rna_z, recon_tcr_z = get_model_predictions(args, 
                                                                                                    G, load_eval, 
                                                                                                    learn_emb_batch_x_rna, learn_emb_batch_tcr)

    tcr_dists = sklearn.metrics.pairwise_distances(preds_tcr, preds_tcr, metric='l1')
    print('Got distances in {:.1f} s'.format(time.time() - t))
    pseudo_tcrs, thresh_fitted = get_pseudoclones(preds_tcr, df.iloc[:, npca:], mask_train, tcr_dists, thresh=args.delta_contrastive)

    print('Done with post-analysis in {:.1f} s!'.format(time.time() - t))


    #######
    #######
    ####### # for k-fold
    #######
    #######

    for patient in patients_holdout:
        patient_mask = df['patient'] == patient
        print('Saving data for patient {}...'.format(patient))
        # with open(os.path.join(args.output_folder, 'preds_patient{}.npz'.format(patient)), 'wb+') as f:
        #     np.savez(f, preds_rna=preds_rna[patient_mask],
        #                 pseudo_tcrs=pseudo_tcrs[patient_mask], tcr_dists=tcr_dists[patient_mask], thresh_fitted=thresh_fitted)
        print('Data saved!')

    #######
    #######
    ####### # for k-fold
    #######
    #######


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
                    learn_emb_batch_x_rna=learn_emb_batch_x_rna.cpu().numpy(), learn_emb_batch_tcr=learn_emb_batch_tcr.cpu().numpy(), learn_emb_batch_labels=learn_emb_batch_labels,
                    mask_train=mask_train.values)
    print('Data saved!')
else:
    print('Skipping post-analysis')
    print('-'*100)

######################################################################
######################################################################
######################################################################





