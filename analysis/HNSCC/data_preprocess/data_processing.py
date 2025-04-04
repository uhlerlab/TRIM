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
from torch import nn
import torch.nn.functional as F
import pickle
fig = plt.figure()



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_folder', default='data_processed')

    args = parser.parse_args()
    return args

args = get_args()

args.output_folder = os.path.join(os.path.expanduser('~'), 'tcr', args.output_folder)
# if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)


def convert_bloodid2num(x):
    vals = np.unique(x)
    vals_dict = {v: i for i,v in enumerate(sorted(vals))}
    output = x.apply(lambda tmp: vals_dict[tmp])
    return output

def library_size_normalize(df, eps=1e-6):
    # log-scale
    # afterwards --> min/max scaling
    df = np.log(df + eps)
    df = df - df.min(axis=1).values[:, np.newaxis]
    df = df / df.max(axis=1).values[:, np.newaxis]

    # df = df / df.sum(axis=1).values[:, np.newaxis] * 1000

    return df

def get_percentiles(col, npctile=10):
    step_size = 100 // npctile
    pctiles = list(range(0, 100 + step_size, step_size))
    bin_edges = []
    for pctile in pctiles:
        bin_edges.append(np.percentile(col, pctile))
    return bin_edges

def get_tcrs(df_blood_metadata, df_tumor_metadata, gaps=True):
    def tcr_parse_cdr3(tmp):
        if pd.notna(tmp) and tmp != 'None':
            tmp = tmp.split('_')
            tmp = [t for t in tmp if t.startswith('TRB')]
            if not tmp:
                return '' 
            else:
                return tmp[0].split('.')[2]
        else:
            return ''

    def tcr_parse_vj(tmp, vj):
        if pd.notna(tmp) and tmp != 'None':
            tmp = tmp.split('_')
            tmp = [t for t in tmp if t.startswith('TRB')]
            if not tmp:
                return ''
            else:
                if vj == 'v':
                    return tmp[0].split('.')[1]
                elif vj == 'j':
                    return tmp[0].split('.')[-1]
                else:
                    raise Exception('must specify v or j')
        else:
            return ''

    def space_string(str_, max_len):
        new_string = str_.strip()
        i_gap = 1
        counter = 0
        while len(new_string) < max_len:
            if counter % 2 == 0:
                new_string = new_string[:i_gap] + ' ' + new_string[i_gap:]
            else:
                new_string = new_string[:len(new_string) - i_gap] + ' ' + new_string[len(new_string) - i_gap:]
            counter += 1
            if counter % 2 == 0:
                i_gap += 2
        return new_string
            
    #########################################################
    ## cdr3 amino acids
    tcrs_blood = df_blood_metadata['vdj'].apply(lambda tmp: tcr_parse_cdr3(tmp))
    tcrs_tumor = df_tumor_metadata['vdj'].apply(lambda tmp: tcr_parse_cdr3(tmp))

    tmpdf1 = pd.DataFrame(np.unique(tcrs_blood[df_blood_metadata['bloodid'] == 'B1'], return_counts=True)).transpose()
    tmpdf1.index = tmpdf1.iloc[:, 0]
    tmpdf1 = tmpdf1.drop(0, axis=1)
    tmpdf1.columns = ['1']

    tmpdf2 = pd.DataFrame(np.unique(tcrs_blood[df_blood_metadata['bloodid'] == 'B2'], return_counts=True)).transpose()
    tmpdf2.index = tmpdf2.iloc[:, 0]
    tmpdf2 = tmpdf2.drop(0, axis=1)
    tmpdf2.columns = ['2']

    tmpdf3 = pd.DataFrame(np.unique(tcrs_tumor[df_tumor_metadata['pre_post'] == 'pre'], return_counts=True)).transpose()
    tmpdf3.index = tmpdf3.iloc[:, 0]
    tmpdf3 = tmpdf3.drop(0, axis=1)
    tmpdf3.columns = ['3']

    tmpdf4 = pd.DataFrame(np.unique(tcrs_tumor[df_tumor_metadata['pre_post'] == 'post'], return_counts=True)).transpose()
    tmpdf4.index = tmpdf4.iloc[:, 0]
    tmpdf4 = tmpdf4.drop(0, axis=1)
    tmpdf4.columns = ['4']

    df_all_tcrs = tmpdf1.join(tmpdf2, how='outer').join(tmpdf3, how='outer').join(tmpdf4, how='outer')
    df_all_tcrs['tmp'] = df_all_tcrs.sum(axis=1)
    df_all_tcrs = df_all_tcrs.sort_values(by='tmp', ascending=False)
    df_all_tcrs = df_all_tcrs.drop('tmp', axis=1)
    df_all_tcrs_index = df_all_tcrs.index.tolist()

    # max_len = max([len(i) for i in df_all_tcrs.index])
    max_len = 80
    if gaps:
        df_all_tcrs.index = [space_string(i, max_len) for i in df_all_tcrs.index]
    else:
        df_all_tcrs.index = [i.ljust(max_len) for i in df_all_tcrs.index]
    

    def helper_lookup(val, lookup):
        try:
            return lookup.index(val)
        except:
           return lookup.index('')


    df_blood_metadata['tcr_index'] = tcrs_blood.apply(lambda tmp: helper_lookup(tmp, df_all_tcrs_index))
    df_tumor_metadata['tcr_index'] = tcrs_tumor.apply(lambda tmp: helper_lookup(tmp, df_all_tcrs_index))

    #########################################################
    ## V/J genes

    df_blood_metadata['tcr_v'] = df_blood_metadata['vdj'].apply(lambda tmp: tcr_parse_vj(tmp, 'v'))
    df_blood_metadata['tcr_j'] = df_blood_metadata['vdj'].apply(lambda tmp: tcr_parse_vj(tmp, 'j'))
    df_tumor_metadata['tcr_v'] = df_tumor_metadata['vdj'].apply(lambda tmp: tcr_parse_vj(tmp, 'v'))
    df_tumor_metadata['tcr_j'] = df_tumor_metadata['vdj'].apply(lambda tmp: tcr_parse_vj(tmp, 'j'))

    all_v_genes = sorted(set(df_blood_metadata['tcr_v'].unique().tolist() + df_tumor_metadata['tcr_v'].unique().tolist()))
    all_j_genes = sorted(set(df_blood_metadata['tcr_j'].unique().tolist() + df_tumor_metadata['tcr_j'].unique().tolist()))

    df_blood_metadata['tcr_v'] = df_blood_metadata['tcr_v'].apply(lambda tmp: all_v_genes.index(tmp))
    df_blood_metadata['tcr_j'] = df_blood_metadata['tcr_j'].apply(lambda tmp: all_j_genes.index(tmp))
    df_tumor_metadata['tcr_v'] = df_tumor_metadata['tcr_v'].apply(lambda tmp: all_v_genes.index(tmp))
    df_tumor_metadata['tcr_j'] = df_tumor_metadata['tcr_j'].apply(lambda tmp: all_j_genes.index(tmp))


    #########################################################

    return df_blood_metadata, df_tumor_metadata, df_all_tcrs, all_v_genes, all_j_genes


########################################################################################################################
############### 
############### START data loading
###############

all_data = []
for fn_ in ['CD4', 'CD8',]:# 'myeloid']:
    print('Loading {}'.format(fn_))
    df_blood_sparse = scipy.io.mmread('/home/che/TRIM/data/Blood{}.mm'.format(fn_)).tocsr()
    df_blood_metadata = pd.read_csv('/home/che/TRIM/data/Blood{}_metadata.csv'.format(fn_), header=0, index_col=0)
    df_blood_cols = pd.read_csv('/home/che/TRIM/data/Blood{}_dimnames1.csv'.format(fn_), header=0, index_col=0)
    df_blood_rows = pd.read_csv('/home/che/TRIM/data/Blood{}_dimnames2.csv'.format(fn_), header=0, index_col=0)


    df_tumor_sparse = scipy.io.mmread('/home/che/TRIM/data/Tumor{}.mm'.format(fn_)).tocsr()
    df_tumor_metadata = pd.read_csv('/home/che/TRIM/data/Tumor{}_metadata.csv'.format(fn_), header=0, index_col=0)
    df_tumor_cols = pd.read_csv('/home/che/TRIM/data/Tumor{}_dimnames1.csv'.format(fn_), header=0, index_col=0)
    df_tumor_rows = pd.read_csv('/home/che/TRIM/data/Tumor{}_dimnames2.csv'.format(fn_), header=0, index_col=0)

    if fn_ == 'myeloid':
        df_blood_metadata['vdj'] = 'myeloid'
        df_tumor_metadata['vdj'] = 'myeloid'

    all_data.append([[df_blood_sparse, df_blood_metadata, df_blood_cols, df_blood_rows],
                     [df_tumor_sparse, df_tumor_metadata, df_tumor_cols, df_tumor_rows]])


df_blood_sparse = scipy.sparse.vstack([tmp[0][0] for tmp in all_data])
df_blood_metadata = pd.concat([tmp[0][1] for tmp in all_data], axis=0)
df_blood_rows = pd.concat([tmp[0][3] for tmp in all_data], axis=0)
df_blood_celltypes = np.concatenate([i * np.ones(tmp[0][0].shape[0]) for i, tmp in enumerate(all_data)])

df_tumor_sparse = scipy.sparse.vstack([tmp[1][0] for tmp in all_data])
df_tumor_metadata = pd.concat([tmp[1][1] for tmp in all_data], axis=0)
df_tumor_rows = pd.concat([tmp[1][3] for tmp in all_data], axis=0)
df_tumor_celltypes = np.concatenate([i * np.ones(tmp[1][0].shape[0]) for i, tmp in enumerate(all_data)])

cols_to_use = sorted(list(set(df_blood_cols.iloc[:, 0]).intersection(set(df_tumor_cols.iloc[:, 0]))))

counts1 = pd.crosstab(df_blood_metadata['patient'], df_blood_metadata['bloodid'])
counts2 = pd.crosstab(df_tumor_metadata['patient'], df_tumor_metadata['pre_post'])
counts = pd.concat([counts1, counts2], axis=1).fillna(0).astype(int)[['B1', 'B2', 'pre', 'post']]


# process tcr strings
df_blood_metadata, df_tumor_metadata, df_all_tcrs, all_v_genes, all_j_genes = get_tcrs(df_blood_metadata, df_tumor_metadata, gaps=False)
vocab = set()
[[vocab.add(c) for c in l] for l in df_all_tcrs.index] 
vocab_char2num = {v: i for i, v in enumerate(sorted(vocab))}
vocab_num2char = {i: v for i, v in enumerate(sorted(vocab))}
df_all_tcrs_array = np.array([[vocab_char2num[char] for char in i] for i in df_all_tcrs.index])
tcr_max_len = df_all_tcrs_array.shape[1]
print(sorted(vocab))



# load each patient separately

# patient_ids = ['P23', 'P24', 'P29', 'P32'] # these have all 4 of: tumor pre/post and blood B1/B2
# Note: we didn't include P27 here since this patient doesn't have blood samples
patient_ids = ['P23', 'P24', 'P29', 'P32', 'P01', 'P02', 'P04', 'P05', 'P08', 'P09', 'P10', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P25', 'P26', 'P28',  'P30', 'P31']
print(counts.loc[patient_ids])

all_df_bb = []
all_df_ba = []
all_df_tb = []
all_df_ta = []
all_celltypes = [[], [], [], []]
all_tcrs = [[], [], [], []]
all_tcr_v = [[], [], [], []]
all_tcr_j = [[], [], [], []]
all_df_metadata = [[], [], [], []]

# for each patient
for i in range(len(patient_ids)):
    # if i > 3: break
    print(i, patient_ids[i])

    ########################################################
    #### get bb
    mask_patient = df_blood_metadata['patient'] == patient_ids[i]
    mask_prepost = df_blood_metadata['bloodid'] == 'B1'
    mask = np.logical_and(mask_patient, mask_prepost)
    rows_to_use = np.argwhere(mask.values)[: ,0].tolist()

    df_bb = df_blood_sparse[rows_to_use, :].todense()
    df_bb = pd.DataFrame(df_bb, index=np.array(df_blood_rows.iloc[:,0].tolist())[mask], columns=df_blood_cols.iloc[:,0].tolist())
    df_bb = df_bb[cols_to_use]

    df_bb = library_size_normalize(df_bb)

    all_celltypes[0].append(df_blood_celltypes[mask])
    all_tcrs[0].append(df_blood_metadata['tcr_index'][mask])
    all_tcr_v[0].append(df_blood_metadata['tcr_v'][mask])
    all_tcr_j[0].append(df_blood_metadata['tcr_j'][mask])
    all_df_metadata[0].append(df_blood_metadata[mask])

    ########################################################
    #### get ba
    mask_patient = df_blood_metadata['patient'] == patient_ids[i]
    mask_prepost = df_blood_metadata['bloodid'] == 'B2'
    mask = np.logical_and(mask_patient, mask_prepost)
    rows_to_use = np.argwhere(mask.values)[: ,0].tolist()

    df_ba = df_blood_sparse[rows_to_use, :].todense()
    df_ba = pd.DataFrame(df_ba, index=np.array(df_blood_rows.iloc[:,0].tolist())[mask], columns=df_blood_cols.iloc[:,0].tolist())
    df_ba = df_ba[cols_to_use]

    df_ba = library_size_normalize(df_ba)

    all_celltypes[1].append(df_blood_celltypes[mask])
    all_tcrs[1].append(df_blood_metadata['tcr_index'][mask])
    all_tcr_v[1].append(df_blood_metadata['tcr_v'][mask])
    all_tcr_j[1].append(df_blood_metadata['tcr_j'][mask])
    all_df_metadata[1].append(df_blood_metadata[mask])

    ########################################################
    #### get tb
    mask_patient = df_tumor_metadata['patient'] == patient_ids[i]
    mask_prepost = df_tumor_metadata['pre_post'] == 'pre'
    mask = np.logical_and(mask_patient, mask_prepost)
    rows_to_use = np.argwhere(mask.values)[: ,0].tolist()

    df_tb = df_tumor_sparse[rows_to_use, :].todense()
    df_tb = pd.DataFrame(df_tb, index=np.array(df_tumor_rows.iloc[:,0].tolist())[mask], columns=df_tumor_cols.iloc[:,0].tolist())
    df_tb = df_tb[cols_to_use]

    df_tb = library_size_normalize(df_tb)

    all_celltypes[2].append(df_tumor_celltypes[mask])
    all_tcrs[2].append(df_tumor_metadata['tcr_index'][mask])
    all_tcr_v[2].append(df_tumor_metadata['tcr_v'][mask])
    all_tcr_j[2].append(df_tumor_metadata['tcr_j'][mask])
    all_df_metadata[2].append(df_tumor_metadata[mask])

    ########################################################
    #### get ta
    mask_patient = df_tumor_metadata['patient'] == patient_ids[i]
    mask_prepost = df_tumor_metadata['pre_post'] == 'post'
    mask = np.logical_and(mask_patient, mask_prepost)
    rows_to_use = np.argwhere(mask.values)[: ,0].tolist()

    df_ta = df_tumor_sparse[rows_to_use, :].todense()
    df_ta = pd.DataFrame(df_ta, index=np.array(df_tumor_rows.iloc[:,0].tolist())[mask], columns=df_tumor_cols.iloc[:,0].tolist())
    df_ta = df_ta[cols_to_use]

    df_ta = library_size_normalize(df_ta)

    all_celltypes[3].append(df_tumor_celltypes[mask])
    all_tcrs[3].append(df_tumor_metadata['tcr_index'][mask])
    all_tcr_v[3].append(df_tumor_metadata['tcr_v'][mask])
    all_tcr_j[3].append(df_tumor_metadata['tcr_j'][mask])
    all_df_metadata[3].append(df_tumor_metadata[mask])

    print(df_bb.shape)
    print(df_ba.shape)
    print(df_tb.shape)
    print(df_ta.shape)

    all_df_bb.append(df_bb)
    all_df_ba.append(df_ba)
    all_df_tb.append(df_tb)
    all_df_ta.append(df_ta)


combined_data = pd.concat([pd.concat(all_df_bb, axis=0), pd.concat(all_df_ba, axis=0), pd.concat(all_df_tb, axis=0), pd.concat(all_df_ta, axis=0)], axis=0)
combined_data_labels_bloodtumor =  np.concatenate([
                                    np.zeros(sum([df.shape[0] for df in all_df_bb])),
                                    np.zeros(sum([df.shape[0] for df in all_df_ba])),
                                    np.ones(sum([df.shape[0] for df in all_df_tb])),
                                    np.ones(sum([df.shape[0] for df in all_df_ta]))
                                    ], axis=0)
combined_data_labels_prepost =  np.concatenate([
                                    np.zeros(sum([df.shape[0] for df in all_df_bb])),
                                    np.ones(sum([df.shape[0] for df in all_df_ba])),
                                    np.zeros(sum([df.shape[0] for df in all_df_tb])),
                                    np.ones(sum([df.shape[0] for df in all_df_ta]))
                                    ], axis=0)
combined_data_labels_celltype = np.concatenate([np.concatenate(all_celltypes[i]) for i in range(len(all_celltypes))], axis=0)
combined_data_labels_patient = np.concatenate([
                                    np.concatenate([i * np.ones(df.shape[0]) for i, df in enumerate(all_df_bb)], axis=0),
                                    np.concatenate([i * np.ones(df.shape[0]) for i, df in enumerate(all_df_ba)], axis=0),
                                    np.concatenate([i * np.ones(df.shape[0]) for i, df in enumerate(all_df_tb)], axis=0),
                                    np.concatenate([i * np.ones(df.shape[0]) for i, df in enumerate(all_df_ta)], axis=0)
                                    ], axis=0)
combined_data_labels_tcr = np.concatenate([np.concatenate(all_tcrs[i]) for i in range(len(all_tcrs))], axis=0)
combined_data_labels_tcr_v = np.concatenate([np.concatenate(all_tcr_v[i]) for i in range(len(all_tcr_v))], axis=0)
combined_data_labels_tcr_j = np.concatenate([np.concatenate(all_tcr_j[i]) for i in range(len(all_tcr_j))], axis=0)

combined_data_labels_metadata = pd.concat([pd.concat(all_df_metadata[0]), pd.concat(all_df_metadata[1]), pd.concat(all_df_metadata[2]), pd.concat(all_df_metadata[3])], axis=0)

combined_data_labels = np.stack([combined_data_labels_bloodtumor,
                                 combined_data_labels_prepost,
                                 combined_data_labels_celltype,
                                 combined_data_labels_patient,
                                 combined_data_labels_tcr,
                                 combined_data_labels_tcr_v,
                                 combined_data_labels_tcr_j,
                                 combined_data_labels_treatment
                                 ], axis=-1)


col_bloodtumor, col_prepost, col_celltype, col_patient, col_tcr, col_tcr_v, col_tcr_j, col_treatment = list(range(combined_data_labels.shape[1]))

############### 
############### END data loading
###############


###############
############### START tcr
############### 
from transformers import BertModel
from transformers import AutoModel
from transformers import FeatureExtractionPipeline
# sys.path.append('/home/mamodio/tcr/tcr-bert/tcr')
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

tcrbert_trb_embedder = load_embed_pipeline("wukevin/tcr-bert", device=0)

s = [' '.join([aa for aa in s.replace(' ', '')]) for s in df_all_tcrs.index]
tcrbert_out = tcrbert_trb_embedder(s)

tcrbert_embeddings = np.stack([np.array(o[0])[0] for o in tcrbert_out])

# pool instead?
# # tcrbert_embeddings = np.stack([np.mean(np.array(o[0]), axis=0) for o in tcrbert_out])
# # tcrbert_embeddings = np.stack([np.mean(np.array(o[0])[1:-1], axis=0) for o in tcrbert_out])

tcrbert_path = '/home/che/TRIM/tcr-bert'
with open(os.path.join(tcrbert_path, 'tcrbert_embeddings.pkl'), 'wb') as f:
    pickle.dump(tcrbert_embeddings, f)
print('tcrbert_embeddings saved to {}'.format(os.path.join(tcrbert_path, 'tcrbert_embeddings.pkl')))

########
########


fn_tcr_embeddings = 'trained_tcr_embeddings_ae.npz'
with open(fn_tcr_embeddings, 'rb') as f:
    tcr_embeddings = np.load(f)['embeddings']

combined_data_tcr = np.take(tcr_embeddings, combined_data_labels[:, col_tcr].astype(np.int32), axis=0)


###############
############### END tcr
############### 




with open(os.path.join(args.output_folder, 'data_rna.npz'), 'wb+') as f:
    np.savez(f, data_rna=combined_data)

with open(os.path.join(args.output_folder, 'data_labels.npz'), 'wb+') as f:
    np.savez(f, data_labels=combined_data_labels)
    
with open(os.path.join(args.output_folder, 'data_tcr.npz'), 'wb+') as f:
    np.savez(f, data_tcr=combined_data_tcr)

with open(os.path.join(args.output_folder, 'data_all_tcrs.npz'), 'wb+') as f:
    np.savez(f, data_all_tcrs=df_all_tcrs.values, rows=df_all_tcrs.index, cols=df_all_tcrs.columns)

with open(os.path.join(args.output_folder, 'combined_data_columns.npz'), 'wb+') as f:
    np.savez(f, cols=combined_data.columns)












