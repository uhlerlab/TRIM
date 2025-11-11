"""
Learn TCR embeddings using an autoencoder.
"""
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
import time
import math
import sklearn.cluster
import logomaker as lm
import os
import pickle
import torch
from torch import nn
import torch.nn.functional as F
fig = plt.figure()

###############
############### Helpers
############### 

class Loader(object):
    """A Loader class for feeding numpy matrices into tensorflow models."""

    def __init__(self, data, labels=None, shuffle=False):
        """Initialize the loader with data and optionally with labels."""
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels] if x is not None]
        self.labels_given = labels is not None

        if shuffle:
            self.r = list(range(data.shape[0]))
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

        if not self.labels_given:  # don't return length-1 list
            return batch[0]
        else:  # return list of data and labels
            return batch

    def iter_batches(self, batch_size=100):
        """Iterate over the entire dataset in batches."""
        num_rows = self.data[0].shape[0]

        end = 0

        if batch_size > num_rows:
            if not self.labels_given:
                yield [x for x in self.data][0]
            else:
                yield [x for x in self.data]
        else:
            for i in range(num_rows // batch_size):
                start = i * batch_size
                end = (i + 1) * batch_size

                if not self.labels_given:
                    yield [x[start:end] for x in self.data][0]
                else:
                    yield [x[start:end] for x in self.data]
            if end < num_rows:
                if not self.labels_given:
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

class TCR_Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.nbase = nbase = kwargs['nbase']
        self.dim_in = dim_in = kwargs["dim_in"]
        self.dimz = dimz = kwargs['dimz']


        # TCR
        self.encoder_tcr = CNN(dim_in=dim_in, dim_out=dimz, dim_len=tcr_max_len, nbase=nbase * 1)
        self.decoder_tcr = CNN_T(dim_in=dimz, dim_out=dim_in, dim_len=tcr_max_len, nbase=nbase * 1)

        # lookups
        self.lookup_vocab = torch.nn.Embedding(len(vocab), n_embedding_vocab).to(device)

        # ops
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        embedding = self.encoder_tcr(x)
        recon = self.decoder_tcr(embedding)
        return embedding, recon

    def safe_lookup_vocab(self, inds):
        inds = inds.to(torch.int32)
        if (inds < 0).any() or (inds > self.lookup_vocab.weight.shape[0]).any():
            raise Exception('trying to lookup outside of vocab range')

        return self.lookup_vocab(inds)

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

        self.bn1 = nn.BatchNorm1d(hdim // 1)
        self.bn2 = nn.BatchNorm1d(hdim // 2)
        self.bn3 = nn.BatchNorm1d(hdim // 4)

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
        self.fc_out1 = nn.Linear(in_features=hdim // 1, out_features=dim_out)

        ksize = 3
        self.conv1 = torch.nn.ConvTranspose1d(hdim // 4, hdim // 2, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose1d(hdim // 2, hdim // 1, kernel_size=ksize, stride=2, padding=(ksize - 1) // 2, output_padding=1)
     

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

        self.layer1 = nn.Linear(in_features=dim_in, out_features=nbase // 1)
        self.layer2 = nn.Linear(in_features=nbase // 1, out_features=nbase // 2)
        self.layer3 = nn.Linear(in_features=nbase // 2, out_features=nbase // 4)
        self.out = nn.Linear(in_features=nbase // 4, out_features=dim_out)

        self.bn1 = nn.BatchNorm1d(nbase // 1)
        self.bn2 = nn.BatchNorm1d(nbase // 2)
        self.bn3 = nn.BatchNorm1d(nbase // 4)

        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        h1 = self.lrelu(self.bn1(self.layer1(x)))
        h2 = self.lrelu(self.bn2(self.layer2(h1)))
        h3 = self.lrelu(self.bn3(self.layer3(h2)))
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

def get_x_tcr_from_label(labels_matrix):
    x_tcr = np.take(get_atchley().values, np.take(df_all_tcrs_array, labels_matrix[:, col_tcr].astype(np.int32), axis=0), axis=0)

    return x_tcr

########################################################################################################################
############### 
############### START data loading
###############

data_path = 'your_data_directory'
pca_file_path = os.path.join(data_path, 'data_rna_pca.pkl')

TRAINING_STEPS = 50100
output_path = os.path.join(data_path, f'tcr_ae/step_{TRAINING_STEPS}')
if not os.path.exists(output_path):
    os.makedirs(output_path)
print('output_path:', output_path)


# Read in data ---------------------------------------------------------
with open(os.path.join(data_path, 'data_rna.pkl'), 'rb') as f:
    data_rna = pickle.load(f)

with open(os.path.join(data_path, 'data_rna_rows.pkl'), 'rb') as f:
    data_rna_rows = pickle.load(f)

with open(os.path.join(data_path, 'data_rna_cols.pkl'), 'rb') as f:
    data_rna_cols = pickle.load(f)

with open(os.path.join(data_path, 'data_labels.pkl'), 'rb') as f:
    data_labels = pickle.load(f)

with open(os.path.join(data_path, 'data_labels_str.pkl'), 'rb') as f:
    data_labels_str = pickle.load(f)

with open(os.path.join(data_path, 'df_all_tcrs.pkl'), 'rb') as f:
    df_all_tcrs = pickle.load(f)

vocab = set()
[[vocab.add(c) for c in l] for l in df_all_tcrs.index] 
vocab_char2num = {v: i for i, v in enumerate(sorted(vocab))}
vocab_num2char = {i: v for i, v in enumerate(sorted(vocab))}
df_all_tcrs_array = np.array([[vocab_char2num[char] for char in i] for i in df_all_tcrs.index])
tcr_max_len = df_all_tcrs_array.shape[1]
print(sorted(vocab))

# Get column indexes
col_bloodtumor = data_labels.columns.get_loc('Tissue')
col_prepost = data_labels.columns.get_loc('Treatment Stage')
col_celltype = data_labels.columns.get_loc('SubCellType')
col_patient = data_labels.columns.get_loc('Patient')
col_tcr = data_labels.columns.get_loc('CDR3(Beta1)')

############### 
############### END data loading
###############

npca = 100
print('PCA to npca:', npca)
if not os.path.exists(pca_file_path):
    pca = sklearn.decomposition.PCA(npca, random_state=0)
    combined_data_pca = pca.fit_transform(data_rna)
    pickle.dump(combined_data_pca, open(pca_file_path, 'wb+'))
    pickle.dump(pca, open(pca_file_path.replace('data_rna_pca.pkl', 'rna_pca.pkl'), 'wb+'))
    print('RNA PCA saved to {}'.format(pca_file_path))
else:
    combined_data_pca = pickle.load(open(pca_file_path, 'rb'))
    print('RNA PCA loaded from {}'.format(pca_file_path))
assert combined_data_pca.shape[1] == npca
assert combined_data_pca.shape[0] == data_rna.shape[0]
print('Finished PCA')

#  use gpu if available
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

x_train = combined_data_pca # note: this is actually not going to be used in learning the tcr embeddings
x_label = data_labels
assert x_train.shape[0] == x_label.shape[0]

# get per-TCR means
print('Before getting per-TCR means x_train.shape:', x_train.shape)
tmp = pd.DataFrame(np.concatenate([x_train, x_label], axis=-1)).groupby(x_train.shape[1] + col_tcr).mean()
print('After groupby mean tmp.shape:', tmp.shape)
x_train = tmp.iloc[:, :x_train.shape[1]].values
print('After getting per-TCR means x_train.shape:', x_train.shape)

tmp2 = tmp.iloc[:, x_train.shape[1]:]
tmp2[tmp.index.name] = tmp.index
tmp2 = tmp2[sorted(tmp2.columns)].values
x_label = tmp2

# training params
dimx = x_train.shape[1]
dimz = 100
batch_size = int(1 * 1024)

n_embedding_vocab = 5

G = TCR_Embedder(dim_in=n_embedding_vocab, dimz=dimz, nbase=int(.5 * 1024))
G = G.to(device)
var_list = [G]


param_list = []
for v in var_list:
    param_list.extend(list(v.parameters()))

opt_G = torch.optim.Adam(param_list, lr=.001)

load_train = Loader(x_train, labels=x_label, shuffle=True)
load_eval = Loader(x_train, labels=x_label, shuffle=False)



i_iter = 0
losses = []
t = time.time()
while i_iter < TRAINING_STEPS:
    i_iter += 1
    [v.train() for v in var_list]
    opt_G.zero_grad()
    batch_loss = []
    
    ########################################################################################################################
    # get data from loader
    batch_x_rna, batch_labels = load_train.next_batch(batch_size)
    batch_x_tcr = get_x_tcr_from_label(batch_labels)

    # convert np to torch objects
    batch_x_rna = numpy2torch(batch_x_rna)
    batch_x_tcr = numpy2torch(batch_x_tcr)
    batch_labels = numpy2torch(batch_labels, type=torch.IntTensor)

    ########################################################################################################################

    batch_embeddings, recon = G(x=batch_x_tcr)

    loss_recon = ((batch_x_tcr - recon)**2).mean()
    batch_loss.append(1 * loss_recon)

    loss_embedding_l2 = (batch_embeddings**2).mean()
    batch_loss.append(.001 * loss_embedding_l2)

    # loss stuff
    batch_loss_list = batch_loss
    batch_loss = torch.mean(torch.stack(batch_loss))
    losses.append(batch_loss.item())

    batch_loss.backward()
    check_for_nan(G)
    opt_G.step()
    opt_G.zero_grad()
    

    if i_iter % 100 == 0:
        print("{:>5}: avg loss: {:.6f} ({:.1f} s)".format(i_iter, np.mean(losses), time.time() - t ))
        if i_iter % 1000 == 0:
            print("{:>5}: batchloss list: {}".format(i_iter, ['{:.6f}'.format(l.detach().cpu().numpy()) for l in batch_loss_list]))
        t = time.time()
        losses = []

    if i_iter % 5000 == 0:
        pass


[v.eval() for v in var_list]
tcr_embeddings = []
for batch_x_rna, batch_labels in load_eval.iter_batches(batch_size):
    batch_x_tcr = get_x_tcr_from_label(batch_labels)

    batch_x_rna = numpy2torch(batch_x_rna)
    batch_x_tcr = numpy2torch(batch_x_tcr)

    # batch_x_tcr = G.safe_lookup_vocab(batch_x_tcr)

    batch_embeddings, batch_recon = G(x=batch_x_tcr)

    tcr_embeddings.append(batch_embeddings.detach().cpu().numpy())

tcr_embeddings = np.concatenate(tcr_embeddings, axis=0)


# Save out embeddings for all TCRs ----------------------------------------------------
if (np.abs(tcr_embeddings).max(axis=0) > 1).sum() < (tcr_embeddings.shape[1] // 2):
    coef = 10
    print("scaling up tcr embeddings by 10x")
else:
    coef = 1
    print("not scaling tcr embeddings")

embeddings_to_save = np.zeros([df_all_tcrs.shape[0], tcr_embeddings.shape[1]])
for i in range(x_label.shape[0]):
    row = int(x_label[i, col_tcr]) # Get the index of the current tcr in df_all_tcrs
    embeddings_to_save[row, :] = coef * tcr_embeddings[i] # Make it so that embeddings_to_save has the same order as df_all_tcrs

fn = f'{output_path}/trained_tcr_embeddings_ae.pkl'
with open(fn, 'wb') as f:
    pickle.dump(embeddings_to_save, f)
print('embeddings_to_save.shape:', embeddings_to_save.shape)
print('Saved out tcr embeddings to', fn)

combined_data_tcr = np.take(embeddings_to_save, data_labels.values[:, col_tcr].astype(np.int32), axis=0)
with open(f'{output_path}/data_tcr.pkl', 'wb') as f:
    pickle.dump(combined_data_tcr, f)
print('combined_data_tcr.shape:', combined_data_tcr.shape)
print('Saved out data_tcr to', f'{output_path}/data_tcr.pkl')


# Some plottings ----------------------------------------------------

print('Plotting some figures...')
umapper_tcr = umap.UMAP(min_dist=.99, n_neighbors=500)
assert tcr_embeddings.shape[0] == x_label.shape[0], "should have same number of tcr embeddings as x_label"
e_tcr_embeddings = umapper_tcr.fit_transform(tcr_embeddings)
print('e_tcr_embeddings.shape:', e_tcr_embeddings.shape)

assert combined_data_tcr.shape[0] == data_labels.shape[0], "should have same number of tcr data as data_labels"
mask_to_plot = np.random.choice(range(combined_data_tcr.shape[0]),min(50000,combined_data_tcr.shape[0]), replace=False)
tcr_embeddings_2 = combined_data_tcr[mask_to_plot]
x_label_2 = data_labels.values[mask_to_plot]
e_tcr_embeddings_2 = umapper_tcr.fit_transform(tcr_embeddings_2)

old_size = fig.get_size_inches()
fig.set_size_inches([6, 3])
length = np.take((df_all_tcrs_array!=0).sum(axis=-1), x_label[:, col_tcr].astype(np.int32))
mask_eval = np.ones(x_label.shape[0], dtype=bool)
r = np.random.choice(range(e_tcr_embeddings.shape[0]), e_tcr_embeddings.shape[0], replace=False)
fig.clf()
ax = fig.subplots(1, 1)
ax.scatter(
    e_tcr_embeddings[r, 0][mask_eval[r]], 
    e_tcr_embeddings[r, 1][mask_eval[r]], 
    s=1, 
    c=length[r][mask_eval[r]])#, vmin=12, vmax=20, cmap=mpl.cm.jet)
ax.set_xticks([])
ax.set_yticks([])
fig.savefig(f'{output_path}/tcr_embedding_color_by_length.png')
fig.set_size_inches(old_size)

cols = [
    (col_celltype, "cell_type"),
    (col_bloodtumor, "tissue"),
    (col_prepost, "prepost"),
    (col_patient, "patient")
]

for col, name in cols:
    old_size = fig.get_size_inches()
    fig.set_size_inches([6, 3])

    labels = x_label_2[:, col]
    unique_labels, color_ids = np.unique(labels, return_inverse=True)
    K = len(unique_labels)
    cmap = plt.get_cmap('tab20', K)

    r = np.random.permutation(e_tcr_embeddings_2.shape[0])

    fig.clf()
    ax = fig.subplots(1, 1)
    ax.scatter(
        e_tcr_embeddings_2[r, 0],
        e_tcr_embeddings_2[r, 1],
        s=2,
        c=color_ids[r],
        cmap=cmap
    )

    ax.set_xticks([])
    ax.set_yticks([])

    # simple legend
    for i, lab in enumerate(unique_labels):
        ax.scatter([], [], color=cmap(i), label=str(lab), s=10)
    ax.legend(title=name, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

    fig.tight_layout()
    fig.savefig(f'{output_path}/tcr_embedding_color_by_{name}.png', dpi=300, bbox_inches='tight')
    fig.set_size_inches(old_size)

old_size = fig.get_size_inches()
fig.set_size_inches([10, 10])
# e = e_rna_emeddings
e = e_tcr_embeddings
epsilon = 10
inds = np.random.choice(range(x_train.shape[0]), 16, replace=False)
fig.clf()
axes = fig.subplots(4, 4)
for i_ax, i in enumerate(inds):
    ax = axes.flatten()[i_ax]

    i + 20
    dists = sklearn.metrics.pairwise_distances(x_train[i][np.newaxis, :], x_train)
    c = dists < epsilon
    colors = np.where(c.ravel(), 'orange', 'lightgray')

    ax.scatter(e[:, 0], e[:, 1], s=1, c=colors)
    ax.scatter(e[i, 0], e[i, 1], s=5, c='r')
[[ax.set_xticks([]), ax.set_yticks([])] for ax in axes.flatten()]
fig.savefig(f'{output_path}/tcr_by_rna_neighbors_eps_{epsilon}.png')
fig.set_size_inches(old_size)


tmp = x_train
mask = (np.triu(np.ones([tmp.shape[0], tmp.shape[0]])) - np.eye(tmp.shape[0])).flatten() == 1

d1 = sklearn.metrics.pairwise_distances(tmp, tmp).flatten()
d2 = sklearn.metrics.pairwise_distances(tcr_embeddings, tcr_embeddings).flatten()
pcc = np.corrcoef(d1[mask][d1[mask] < epsilon], d2[mask][d1[mask] < epsilon])[0, 1]
print("r = {:.3f}".format(pcc))


r = np.random.choice(range(d1[mask].shape[0]), 5000, replace=False)
fig.clf()
ax = fig.subplots(1, 1)
ax.scatter(d1[mask][r], d2[mask][r], s=1)
[ax.set_xticks([]), ax.set_yticks([])]
ax.set_title("r = {:.3f}".format(pcc))
fig.savefig(f'{output_path}/distance_between_rna_tcr_emb.png')

print('Done!')













