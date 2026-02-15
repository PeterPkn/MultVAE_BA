import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import numpy as np
import os

class MovieLensDataset(Dataset):
    def __init__(self, data_dir, split='train', fold_id=0):
        self.data = sp.load_npz(os.path.join(data_dir, f"{fold_id}", f'{split}_input.npz'))
        self.num_users, self.num_items = self.data.shape
        self.num_interactions = self.data.sum()

        if split == 'train':
            # During training, we want to recreate the input
            self.targets = self.data
        else:
            self.targets = sp.load_npz(
                os.path.join(data_dir, f"{fold_id}", f'{split}_target.npz')
            )
        if not split == 'train':
            self.num_interactions += self.targets.sum()

        self.__ensure_types()

    def __len__(self):
        return self.num_users
    
    def __ensure_types(self):
        self.data = self.data.astype("float32")
        self.targets = self.targets.astype("float32")

    def __getitem__(self, idx):
        x_sample = self.data[idx].toarray().squeeze()
        y_sample = self.targets[idx].toarray().squeeze()
        
        return x_sample, y_sample
    
def get_ml1m_dataloaders():
    train_fold_list = []
    for i in range(5):
        train_fold_list.append(
            MovieLensDataset(
                data_dir=os.path.join('datasets', 'processed', 'ml-1m'),
                split='train',
                fold_id=i
            )
        )
    val_fold_list = []
    for i in range(5):
        val_fold_list.append(
            MovieLensDataset(
                data_dir=os.path.join('datasets', 'processed', 'ml-1m'),
                split='val',
                fold_id=i
            )
        )
    test_fold_list = []
    for i in range(5):
        test_fold_list.append(
            MovieLensDataset(
                data_dir=os.path.join('datasets', 'processed', 'ml-1m'),
                split='test',
                fold_id=i
            )
        )
    return train_fold_list, val_fold_list, test_fold_list