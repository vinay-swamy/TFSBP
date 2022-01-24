
import torch
from torch.utils.data import Dataset 
import pickle
import numpy as np


class AaPwmDataset(Dataset):
    def __init__(self, pickle_file, aa_mat_size, pwm_mat_size):
        with open(pickle_file, 'rb') as instream:
            dict_list = pickle.load(instream)
        self.AA_PWM_dict_list = dict_list
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
