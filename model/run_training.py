#%%
import torch
from trainer import Trainer
from model import BasicTFBSPredictor
from torch import nn 

AA_MAT_SIZE = 896
PWM_MAT_SIZE=19

tfbs_model = BasicTFBSPredictor(
    AA_mat_size = AA_MAT_SIZE, 
    PWM_mat_size = PWM_MAT_SIZE

)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(tfbs_model.parameters(), lr = .001)

#%%
trainer_instance = Trainer(
    model=tfbs_model, 
    pickle_file= "/Users/vinayswamy/columbia/aqlab/TFBindSeqPred/all_jaspar_pwm_with_aa_seq.pickle", 
    metadata_file = "/Users/vinayswamy/columbia/aqlab/TFBindSeqPred/jaspar_protein_metadata.csv", 
    batch_size=500,
    aa_mat_size = AA_MAT_SIZE, 
    pwm_mat_size = PWM_MAT_SIZE,
    loss_fn = loss_fn, 
    optimizer = optimizer , 
    n_epochs = 10, 
    device = 'cpu', 
    seed = 
    110101
)
