#%%
from matplotlib.pyplot import get
import torch
from torch import nn
from AA_PWM_dataloader import AaPwmDataset

def lo(lin, padding, dilation, ks, stride):
    top = lin + 2*padding - dilation*(ks -1) - 1
    res = top/stride + 1
    return res 

def get_conv_kernel_size(Li, Lo, padding, dilation):
    ks =((Lo-Li-2)*-1) +1
    return int(ks)
#%%
class BasicTFBSPredictor(nn.Module):
    def __init__(self, AA_mat_size, PWM_mat_size):
        super(BasicTFBSPredictor, self).__init__()
        self.PWM_mat_size = PWM_mat_size
        self.AA_mat_size = AA_mat_size
        ## (batch, 20, AA_MAT_SIZE)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=20, 
                      out_channels=40, 
                      kernel_size = 4,
                      stride = 2,
                      padding=1
                      ),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.MaxPool1d(2, 2),
             ## (batch, 20, AA_MAT_SIZE/ 4)
            nn.Conv1d(in_channels=40, 
                      out_channels=80, 
                      kernel_size= 4,
                      stride = 2,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.MaxPool1d(kernel_size = 2, stride = 2)
        )
         ## (batch, 20, AA_MAT_SIZE/ 16)
        self.dense_block =nn.Sequential(
            nn.Linear(in_features=int(AA_mat_size/16 * 80 ),
                     out_features=1024),
            nn.ReLU(),
            nn.Dropout(.3), 
            nn.Linear(in_features=1024,
                      out_features=256),
                                  nn.ReLU(),
            nn.Dropout(.3), 
            nn.Linear(in_features=256, 
                      out_features= 4 * self.PWM_mat_size)

        )

       
    def forward(self, x):

        x=self.conv_block(x)     
        x=torch.flatten(x, 1)       
        x=self.dense_block(x)
        x=torch.reshape(x, (-1,  4,self.PWM_mat_size))
        return x

#%%
