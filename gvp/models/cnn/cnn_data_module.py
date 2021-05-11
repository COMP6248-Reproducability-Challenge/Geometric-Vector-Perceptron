from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl

import gvp

def scaling(y):
    min_y = y.min() #-10.
    max_y = y.max() #10.
    return ( y - min_y ) / ( max_y - min_y )

class CNNDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, task, num_workers=8, scaling=False):
        # 0: off-center, 1: perimeter 2: combined
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.task = task
        self.num_workers = num_workers
        self.scaling = scaling

    def setup(self, stage=None):
        taskmap = {'off_center':0,
                   'perimeter':1,
                   'combined':2
                   }
        X = torch.Tensor( np.load(self.data_dir/"cnn.npy") ).permute(0,4,1,2,3)
        # X = X[:, ::2] # skip channel 1 - non-special atoms
        
        y = []
        with np.load(self.data_dir/"answers.npz") as f:
            if self.scaling:
                y.append( scaling(torch.Tensor(f["off_center"])) )
                y.append( scaling(torch.Tensor(f["perimeter"])) )
                y.append( torch.abs(y[0]-y[1]) )
            else:
                y.append( torch.Tensor(f["off_center"]) )
                y.append( torch.Tensor(f["perimeter"]) )
                y.append( torch.abs(scaling(y[0]) - scaling(y[1])) )

        dataset = TensorDataset(X, y[taskmap[self.task]])

        # Split
        full, test = random_split(dataset, [18000, 2000])

        if stage == 'fit' or stage is None:
            self.cnn_train, self.cnn_val = random_split(full, [16000, 2000])
            self.dims = tuple(self.cnn_train[0][0].shape)
        if stage == 'test' or stage is None:
            self.cnn_test = test
            self.dims = tuple(self.cnn_test[0][0].shape)
    
    def train_dataloader(self):
        return DataLoader(self.cnn_train, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def val_dataloader(self):
        return DataLoader(self.cnn_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cnn_test, batch_size=self.batch_size, num_workers=self.num_workers)