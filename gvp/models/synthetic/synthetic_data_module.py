import gvp
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, task, transform, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.task = task
        self.transform = transform
        self.num_workers = num_workers

    def setup(self, stage):
        synthetic = torch.from_numpy(np.load(self.data_dir/"synthetic.npy"))
        with np.load(self.data_dir/"answers.npz") as data:
            targets = torch.from_numpy(data[self.task])

        num_structs = targets.shape[0] # number of stuctures
        
        # add one-hot vector to the last dimension
        is_special = torch.zeros((20000, 2, 100, 1))
        is_special[:, 1, :3] = 1 
        synthetic = torch.cat([synthetic, is_special], dim=3) # (2000, 2, 100, 3) -> (20000, 2, 100, 4) -- last channel corresponds to is_special

        data_list = [self.transform(Data(x=synthetic[n, 1], pos=synthetic[n, 0, :, :3], norm=synthetic[n, 1, :, :3], y=targets[n])) for n in range(num_structs)]


        self.train_set, self.test_set = train_test_split(
            data_list, test_size=0.1
        )
        self.train_set, self.val_set = train_test_split(
            self.train_set, test_size=0.1
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)