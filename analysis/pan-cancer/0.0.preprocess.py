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
df_metadata = pd.read_csv(f'{data_path}/GSE156728_metadata.txt', sep='\t', index_col=0)
df_tcr = pd.read_csv(f'{data_path}/GSE156728_10X_VDJ.merge.txt', sep='\t', index_col=0)

perform_pca = False
if perform_pca:
    print('Starting loading')
    t = time.time()
    # fns = sorted(glob.glob('./*'))
    fns = sorted(glob.glob(f'{data_path}/*'))
    fns = [fn for fn in fns if ('CD4' in fn) or ('CD8' in fn)]
    df_list = []
    for i, fn in enumerate(fns):
        print(i)
        df_tmp = pd.read_csv(fn, sep='\t', index_col=0).T
        df_tmp = library_size_normalize(df_tmp)
        if 'CD4' in fn:
            df_tmp['cellType'] = 'CD4'
        elif 'CD8' in fn:
            df_tmp['cellType'] = 'CD8'
        print(df_tmp.shape)
        print('')
        df_list.append(df_tmp)
    print('{:.1f}'.format(time.time() - t))

    print('Starting concat')
    t = time.time()
    cols_list = [t.columns.tolist() for t in df_list]
    cols_set = set(cols_list[0])
    for cols in cols_list:
        cols_set = cols_set.intersection(cols)
    cols_set = list(cols_set)

    df = pd.concat([df[cols_set].fillna(0) for df in df_list])
    print('{:.1f} s\n'.format(time.time() - t))

    # Save cellType labels
    df_cellType = df[['cellType']]
    # df_cellType.to_csv('/data/che/panc/data_cellType.csv')
    df = df.drop(columns=['cellType'])

    print('Starting PCA')
    t = time.time()
    pca = sklearn.decomposition.PCA(100)
    pca_data = pca.fit_transform(df.values)
    df = pd.DataFrame(pca_data, index=df.index)
    print('{:.1f} s\n'.format(time.time() - t))

    print('Starting merging')
    t = time.time()
    df_combined = df.merge(df_metadata, how='left', left_index=True, right_index=True)
    df_combined = df_combined.merge(df_tcr[~df_tcr.index.duplicated()], how='left', left_index=True, right_index=True)

    df_combined['cdr3'] = df_combined['cdr3'].fillna('None')
    print('{:.1f} s\n'.format(time.time() - t))


    print('Starting to save...')
    t = time.time()
    with open('/data/che/panc/data_pca.npz', 'wb+') as f:
        np.savez(f, df=df_combined.values, df_index=df_combined.index, df_columns=df_combined.columns)
    print('{:.1f} s\n'.format(time.time() - t))
else:
    print('Skipping PCA analyses')
    print('-'*100)

######################################################################
######################################################################
######################################################################

######################################################################
######################################################################
######################################################################
# CNN autoencoder for TCRs

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
        # x = self.safe_lookup_vocab(x)

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
        self.conv3 = torch.nn.Conv1d(hdim // 2, hdim // 4, kernel_size=ksize, stride=strides[2], padding=(ksize - 1) // 2)#'same')

        self.bn1 = nn.BatchNorm1d(hdim // 1) # nn.Identity() # 
        self.bn2 = nn.BatchNorm1d(hdim // 2) # nn.Identity() # 
        self.bn3 = nn.BatchNorm1d(hdim // 4) # nn.Identity() # 

        sum_of_strides = sum([s > 1 for s in strides])
        self.fc_out = nn.Linear(in_features=(hdim // 4) * math.ceil(dim_len / (2**sum_of_strides)), out_features=dim_out)
        # self.fc_out = nn.Linear(in_features=(hdim // 4) * (dim_len // (2**sum([s > 1 for s in strides]) + 1)), out_features=dim_out)

        self.lrelu = torch.nn.LeakyReLU()

        self.first = True

    def forward(self, x):
        x = x.permute([0, 2, 1])

        h1 = self.lrelu(self.bn1(self.conv1(x)))
        h2 = self.lrelu(self.bn2(self.conv2(h1)))
        h3 = self.lrelu(self.bn3(self.conv3(h2)))
        h3_flat = h3.view([x.shape[0], -1])

        out = self.fc_out(h3_flat)
        # out = torch.mean(h3, axis=-1)

        if self.first:
            print(h1.shape)
            print(h2.shape)
            print(h3.shape)
            # print(h3_flat.shape)
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

        # out = self.softmax(    self.bn_out(self.fc_out1(out).permute([0,2,1])).permute([0,2,1])   )
        out = self.fc_out1(out)

        # out = self.bn_out(self.conv_out(out))
        # out = out[:, : ,:self.dim_len]
        # out = out.permute([0, 2, 1])
        # out = self.softmax(out)
        # out = torch.sigmoid(out)
        # out = out.reshape([out.shape[0], -1])

        return out

def numpy2torch(x, type=torch.FloatTensor):

    return torch.from_numpy(x).type(type).to(device)

def one_hot_along_3rd_axis(x):
    out = np.zeros([x.shape[0], x.shape[1], len(vocab)])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j, x[i, j]] = 1
    return out


npca = 100

with open(f'{data_path}/data_pca.npz', 'rb') as f:
    npzfile = np.load(f, allow_pickle=True)
    df = pd.DataFrame(npzfile['df'], index=npzfile['df_index'], columns=npzfile['df_columns'])

mask = df['cdr3'] != 'None'
# mask = np.logical_and(mask, df['patient'].apply(lambda tmp: tmp in ['L.P20190404', 'CHOL.P0216', 'MM.P20190322']))
df = df[mask]

df_tcr_counts = df['cdr3'].value_counts()
df['cdr3_count'] = np.array([df_tcr_counts.loc[tcr] for tcr in df['cdr3'].tolist()])

df_all_tcrs = pd.DataFrame(df['cdr3'].unique())
tcr_max_len = max(df_all_tcrs.iloc[:, 0].apply(lambda tmp: len(tmp)))
df_all_tcrs.iloc[:, 0] = df_all_tcrs.iloc[:, 0].apply(lambda tmp: tmp.ljust(tcr_max_len))

vocab = set()
[[vocab.add(c) for c in l] for l in df_all_tcrs.iloc[:, 0]]
vocab_char2num = {v: i for i, v in enumerate(sorted(vocab))}
vocab_num2char = {i: v for i, v in enumerate(sorted(vocab))}
df_all_tcrs_array = np.array([[vocab_char2num[char] for char in i] for i in df_all_tcrs.iloc[:, 0]])
tcr_max_len = df_all_tcrs_array.shape[1]
print(sorted(vocab))
df_all_tcrs_array = one_hot_along_3rd_axis(df_all_tcrs_array)

train_tcr_ae = False

if train_tcr_ae:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAINING_STEPS = 10100
    dimx = df_all_tcrs_array.shape[1]
    dimz = 100
    batch_size = int(1 * 1024)

    n_embedding_vocab = df_all_tcrs_array.shape[2]



    G = TCR_Embedder(dim_in=n_embedding_vocab, dimz=dimz, nbase=int(.05 * 1024))
    G = G.to(device)
    var_list = [G]

    param_list = []
    for v in var_list:
        param_list.extend(list(v.parameters()))

    opt_G = torch.optim.Adam(param_list, lr=.001)

    load_train = Loader(df_all_tcrs_array, shuffle=True)
    load_eval = Loader(df_all_tcrs_array, shuffle=False)


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
        batch_x_tcr = load_train.next_batch(batch_size)

        # convert np to torch objects
        batch_x_tcr = numpy2torch(batch_x_tcr)
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
        opt_G.step()
        opt_G.zero_grad()
        
        if i_iter % 100 == 0:
            print("{:>5}: avg loss: {:.5f} ({:.1f} s)".format(i_iter, np.mean(losses), time.time() - t ))
            if i_iter % 1000 == 0:
                print("{:>5}: batchloss list: {}".format(i_iter, ['{:.3f}'.format(l.detach().cpu().numpy()) for l in batch_loss_list]))
            t = time.time()
            losses = []




    [v.eval() for v in var_list]
    tcr_embeddings = []
    for batch_x_tcr in load_eval.iter_batches(batch_size):
        batch_x_tcr = numpy2torch(batch_x_tcr)

        batch_embeddings, _ = G(x=batch_x_tcr)

        tcr_embeddings.append(batch_embeddings.detach().cpu().numpy())

    tcr_embeddings = np.concatenate(tcr_embeddings, axis=0)





    embeddings_to_save = []
    df_all_tcrs_stripped = [s.strip() for s in df_all_tcrs.iloc[:, 0]]
    for cdr3 in df['cdr3']:
        ind = df_all_tcrs_stripped.index(cdr3)
        embeddings_to_save.append(tcr_embeddings[ind])
    embeddings_to_save = np.stack(embeddings_to_save)

    fn_embeddings = f'{data_path}/trained_tcr_embeddings_ae.npz'
    with open(fn_embeddings, 'wb+') as f:
        np.savez(fn_embeddings, embeddings=embeddings_to_save)
else:
    print('Skipping training of TCR autoencoder')
    print('-'*100)


######################################################################
######################################################################
######################################################################
