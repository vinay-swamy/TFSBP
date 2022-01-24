from curses import meta
import torch 
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle

from AA_PWM_dataloader import AaPwmDataset
class Trainer:
    def __init__(self, 
                 model, 
                 pickle_file, 
                 metadata_file, 
                 batch_size,
                 aa_mat_size, 
                 pwm_mat_size,
                 loss_fn, 
                 optimizer, 
                 n_epochs, 
                 device, 
                 seed):
        
        with open(pickle_file, 'rb') as instream:
            dict_list = pickle.load(instream)
        
        metadata = pd.read_csv(metadata_file)
        
        #80/10/10  
        train_metadata = metadata.sample(frac = .8,random_state=seed)
        non_train_metadata = metadata[~metadata.uniprot_id.isin(train_metadata['uniprot_id'])]
        validation_metadata = non_train_metadata.sample(frac = .5, random_state = seed)
        test_metadata = non_train_metadata[~non_train_metadata.uniprot_id.isin(validation_metadata['uniprot_id'])]


        self.training_data = DataLoader(dataset =  AaPwmDataset(metadata=train_metadata, 
                                                     dict_list = dict_list, 
                                                     aa_mat_size = aa_mat_size, 
                                                     pwm_mat_size = pwm_mat_size
                                                     ), 
                                        batch_size=batch_size
                                                        )
        self.validation_data = DataLoader(dataset =  AaPwmDataset(metadata=validation_metadata, 
                                                     dict_list = dict_list, 
                                                     aa_mat_size = aa_mat_size, 
                                                     pwm_mat_size = pwm_mat_size
                                                     ), 
                                        batch_size=batch_size
                                                        )
        
        self.test_data = DataLoader(dataset =  AaPwmDataset(metadata=test_metadata, 
                                                     dict_list = dict_list, 
                                                     aa_mat_size = aa_mat_size, 
                                                     pwm_mat_size = pwm_mat_size
                                                     ), 
                                        batch_size=batch_size
                                                        )
        
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n_epochs=n_epochs
        self.device =device
        

    def train(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()
        n_batches = len(dataloader)
        all_loss = np.repeat([None],n_batches )
        for batch_num, batch in enumerate(dataloader):
            feature, label = batch 
            feature = feature.to(self.device)
            label = label.to(self.device)
            pred = self.model(feature)
            loss = self.loss_fn(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_loss[batch_num] = loss.item()
        
        return np.mean(all_loss)

    def test(self, dataloader):
        test_size = len(dataloader.dataset)
        n_batches = len(dataloader)
        all_loss = np.repeat([None],n_batches )
        with torch.no_grad():
            for batch_num, batch in enumerate(self.dataloader):
                feature, label = batch 
                feature = feature.to(self.device)
                label = label.to(self.device)
                pred = self.model(feature)
                loss = self.loss_fn(pred, label)
                all_loss[batch_num] = loss.item()
        return np.mean(all_loss)
    
    def train_loop(self):
        all_training_loss = []
        all_validation_loss = []
        for e in range(self.n_epochs):
            train_loss = self.train()
            all_training_loss.append(train_loss)
            validation_loss = self.test()
            all_validation_loss.append(validation_loss)
        return pd.DataFrame({'epoch' : list(range(self.n_epochs)), 
                             'train_loss' : all_training_loss,
                             'validation_loss' : all_validation_loss
                             })








