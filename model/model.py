#%%
from matplotlib.pyplot import get
import torch
from torch import nn

def lo(lin, padding, dilation, ks, stride):
    top = lin + 2*padding - dilation*(ks -1) - 1
    res = top/stride + 1
    return res 

def get_conv_kernel_size(Li, Lo, padding, dilation):
    ks =((Lo-Li-2)*-1) +1
    return int(ks)
#%%
get_conv_kernel_size(896, 896/2, 1, 1)
#lo(896, 1, 1,451, 1 )
#%%

class BasicTFBSPredictor(nn.Module):
    def __init__(self, AA_mat_size, PWM_mat_size):
        super(BasicTFBSPredictor, self).__init__()
        self.PWM_mat_size = PWM_mat_size
        self.AA_mat_size = AA_mat_size

        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channels=20, 
                      out_channels=8, 
                      kernel_size = get_conv_kernel_size(AA_mat_size, AA_mat_size/2, 1, 1),
                      padding=1
                      ),
            nn.ReLU(),
            nn.Dropout(.3)
        )   
        self.second_conv =nn.Sequential(
            nn.Conv1d(in_channels=8, 
                      out_channels=4, 
                      kernel_size= get_conv_kernel_size(AA_mat_size/2, AA_mat_size/4, 1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Dropout(.3)
        )
        self.last_conv = nn.Sequential(
            nn.Conv1d(in_channels=4, 
                      out_channels=2, 
                      kernel_size= get_conv_kernel_size(AA_mat_size/4, AA_mat_size/8, 1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Dropout(.3)
        )
        self.first_dense =nn.Sequential(
            nn.Linear(in_features=int(AA_mat_size/4 ),
                     out_features=4*PWM_mat_size),
            nn.ReLU(),
            nn.Dropout(.3)
        )

        self.final_dense = nn.Linear(in_features=4*PWM_mat_size, out_features=4*PWM_mat_size)
        self.sm = nn.Softmax(dim = 1)
    def forward(self, x):

        x=self.first_conv(x)
        print(x.shape)
        x=self.second_conv(x)
        print(x.shape)
        x=self.last_conv(x)
        print(x.shape)
        x=torch.flatten(x, 1)
        print(x.shape)
        x=self.first_dense(x)
        print(x.shape)
        x=self.final_dense(x)
        print(x.shape)
        x=torch.reshape(x, (-1,  4,self.PWM_mat_size))
        x=self.sm(x)
        return x

# %%
M = BasicTFBSPredictor(896,19)
# %%
k= torch.randn((10, 20, 896))
# %%
x=M(k)
# %%
x.shape
# %%
torch.sum(x[0,:,:],0 )
# %%
