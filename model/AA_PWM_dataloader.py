
import torch
from torch.utils.data import Dataset 
import numpy as np
import pandas as pd

def torch2icdf(x):
    x=x.numpy()
    x=np.log2(x+1)
    x=np.sum(x, 0)
    x=1/x
    x[x==np.inf]=0
    return pd.DataFrame({'x':list(range(len(x))), 'y': x })

class AaPwmDataset(Dataset):
    def __init__(self, metadata, dict_list, aa_mat_size, pwm_mat_size):
        
        valid_ids = set(metadata['uniprot_id'].to_list())
        self.AA_PWM_dict_list = [x for x in dict_list if x['uniprot_id'] in valid_ids]
        if len(self.AA_PWM_dict_list) != len(valid_ids):
            print("Warning: Not all IDs in metadata matched to dict list")
        self.aa_mat_size = aa_mat_size 
        self.pwm_mat_size = pwm_mat_size
        self.pwm_center = int(self.pwm_mat_size/2)
    def __len__(self):
        return len(self.AA_PWM_dict_list)
    def __getitem__(self, index):
        pwm_processed = self.center_pad_pwm(self.AA_PWM_dict_list[index]['pwm'])
        aa_mat_processed = self.pad_aa_seqmat(self.AA_PWM_dict_list[index]['aa_seq_mat'])
        return torch.tensor(aa_mat_processed, dtype=torch.float32), torch.tensor(pwm_processed, dtype=torch.float32)
    def center_pad_pwm(self, mat):
        ## fix case where PWM > pad 
        mat_ic = np.log2(mat +1)
        total_ic = np.sum(mat_ic, 0)
        most_ic = np.min(total_ic)   
        which_most_ic = np.where(total_ic== most_ic)[0]
        center = mat.shape[1]/ 2
        closest_to_center = np.argmin(np.abs(which_most_ic - center))
        min_ic_idx = which_most_ic[closest_to_center]
        padded_pwm = np.zeros((4, self.pwm_mat_size))
        start = self.pwm_center - min_ic_idx
        end = self.pwm_center + (mat.shape[1]- min_ic_idx)
        if start < 0 or end > self.pwm_mat_size:
            trim_right = min(min_ic_idx + self.pwm_center, mat.shape[1] )
            trim_left = max(min_ic_idx - self.pwm_center, 0 )
            newmat = mat[:,trim_left:trim_right]
            return self.center_pad_pwm(newmat)
        padded_pwm[:, start:end] = mat
        return padded_pwm
    def pad_aa_seqmat(self, mat):
        padded_mat = np.zeros((mat.shape[0], self.aa_mat_size))
        rhs=min([mat.shape[1], self.aa_mat_size])
        padded_mat[:,:rhs]=mat[:,:rhs]
        return padded_mat
    def plot_ic(self, aggfn):
        all_ic = [ torch2icdf(self.__getitem__(i)[1]) for i in range(self.__len__()) ]
        all_ic = pd.concat(all_ic)     
        summed_ic = all_ic.groupby('x').agg(aggfn).reset_index(drop=False)
        import plotnine as pn
        p=(
            pn.ggplot(summed_ic) +
                pn.geom_col(pn.aes(x='x', y='y')) +
                pn.xlab('centered PWM position') + 
                pn.ylab('Aggregated Information content\n(IC =1/log2(p(nt)))') +
                pn.theme_bw()

        )
        print(p)
        return