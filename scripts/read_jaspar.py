#%%
import glob
import os 
import numpy as np 
import re
import pandas as pd
from Bio import SeqIO
import pickle 
DIGITS=re.compile(r"\d+")
os.chdir('/Users/vinayswamy/columbia/aqlab/TFBindSeqPred')
def read_jaspar(filepath):
    with open(filepath) as instream:
        header = nextline=instream.readline()[1:].split('\t')[0].split(".")[0]
        pwm_list=[DIGITS.findall( instream.readline()) for _ in range(4)]
    pwm = np.asarray(pwm_list, dtype=np.float32) 
    scale_factor = np.sum(pwm[:,0]).astype(np.int64)
    pwm = pwm / scale_factor
    return (header, pwm)


AA_MAP = { "A":0,
    "C":1, 
    "D":2,
    "E":3, 
    "F":4,
    "G":5,
    "H":6,
    "I":7, 
    "K":8, 
    "L":9,
    "M":10,
    "N":11, 
    "P":12, 
    "Q":13,
    "R":14,
    "S":15,
    "T":16,
    "V":17, 
    "W":18,
    "Y":19 
}

NT_MAP = {
"A":0,
"C":1, 
"G":2,
"T":3
}

def encode_seq(seq, map):
    empty = np.zeros((len(map), len(seq) ))
    for i in range(len(seq)):
        monomer = seq[i]
        idx = map[monomer]
        empty[idx, i]=1
    return empty

# %%
jfiles = glob.glob("jaspar_pfm/*.jaspar")
pwm_iter = (read_jaspar(j) for j in jfiles)

# %%
all_pwms = {header:pwm for header, pwm in pwm_iter}
pwm_jids = set(list(all_pwms.keys()))
len(pwm_jids) # 1997 distinct bs sequences 
# %%
jaspar_self_ids = pd.read_csv("jaspar_self_ids.txt", sep = "|", 
names=['self_id', "collection", "jaspar_id", "nn", "gene_name"]).query('collection == "CORE"').query('nn == 1')
## not sure what the nn column means; have the same gene name and jaspar_id so foing to assume they are the same 

sum(jaspar_self_ids.jaspar_id.isin(pwm_jids))#1996/1997 detected. only MA0835 is missing

#%%
jaspar_up_ids = pd.read_csv("jaspar_uniprot_ids.txt", 
names = ['self_id', 'uniprot_id'], sep = "|")

jaspar_ids = jaspar_self_ids.merge(jaspar_up_ids, how='left')
sum(jaspar_ids.uniprot_id.isna()) # missing 60 ids. looks like there's an issue related to jaspar id version

jaspar_ids_missing_up_id = jaspar_ids[jaspar_ids.uniprot_id.isna()]

jaspar_ids = jaspar_ids[~jaspar_ids.uniprot_id.isna()]

#%%
jaspar_ids[jaspar_ids.duplicated('uniprot_id', keep=False)]
jaspar_ids_dup_up_id = jaspar_ids[jaspar_ids.duplicated('uniprot_id', keep='first')]
jaspar_ids = jaspar_ids.drop_duplicates('uniprot_id',keep='first' ).set_index('uniprot_id')
# 1802 valid binding sites 
up_ids = set(jaspar_ids.index.to_list())
len(up_ids)
# %%
fa = SeqIO.parse("uniprot_sprot.fasta", "fasta")

seq_mat_dict = []
for record in fa:
    t_up_id = record.id.split("|")[1]
    if  t_up_id in up_ids:
        onehot_seq_mat = encode_seq(record.seq, AA_MAP)
        t_jasp_id = jaspar_ids.loc[t_up_id, "jaspar_id"]
        pwm = all_pwms[t_jasp_id]
        seq_mat_dict.append( { 'uniprot_id' : t_up_id, 
          'jaspar_id' : t_jasp_id,
          'aa_seq_str' : str(record.seq), 
          'aa_seq_mat' : onehot_seq_mat, 
          'pwm' :pwm }
        )
#%%
available_up_id = [x['uniprot_id'] for x in seq_mat_dict ]
filtered_metadata = jaspar_ids.reset_index(drop=False).pipe(lambda x: x[x.uniprot_id.isin(available_up_id)])
filtered_metadata.to_csv('jaspar_protein_metadata.csv',index=False)
# %%
with open('all_jaspar_pwm_with_aa_seq.pickle', 'wb+') as outstream:
    pickle.dump(seq_mat_dict, outstream)
#%%



# %%
seq_mat_dict
protein_lens = np.asarray([x['aa_seq_mat'].shape[1] for x in seq_mat_dict ])
## use 896 as the aa mat window size
bs_lens = np.asarray([x['pwm'].shape[1] for x in seq_mat_dict ])
## use 19 as the PWM window size.
# %%# %%
