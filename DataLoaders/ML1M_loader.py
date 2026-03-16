import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import pandas as pd
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, split='train', fold_id=0, global_indexing = False):
        self.global_indexing = global_indexing
        self.data = sp.load_npz(os.path.join(data_dir, f"{fold_id}", f'{split}_input.npz'))
        self.num_users, self.num_items = self.data.shape
        self.num_interactions = self.data.sum()
        self.id_offset = fold_id * self.num_users

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
        
        return x_sample, y_sample, idx + self.id_offset if self.global_indexing else idx  # Return the global user index for this fold
    
def get_dataset_dataloaders(global_indexing=False, dataset='ml-1m'):
    train_fold_list = []
    train_user_info = []
    val_user_info = []
    test_user_info = []
    val_fold_list = []
    test_fold_list = []

    for i in range(5):
        train_fold_list.append(
            CustomDataset(
                data_dir=os.path.join('datasets', 'processed', dataset),
                split='train',
                fold_id=i,
                global_indexing=global_indexing
            )
        )
        val_fold_list.append(
            CustomDataset(
                data_dir=os.path.join('datasets', 'processed', dataset),
                split='val',
                fold_id=i,
                global_indexing=global_indexing
            )
        )
        test_fold_list.append(
            CustomDataset(
                data_dir=os.path.join('datasets', 'processed', dataset),
                split='test',
                fold_id=i,
                global_indexing=global_indexing
            )
        )
        train_user_info.append(pd.read_csv(os.path.join(os.path.join('datasets', 'processed', dataset), f"{i}", f'train_user_info.csv')))
        val_user_info.append(pd.read_csv(os.path.join(os.path.join('datasets', 'processed', dataset), f"{i}", f'val_user_info.csv')))
        test_user_info.append(pd.read_csv(os.path.join(os.path.join('datasets', 'processed', dataset), f"{i}", f'test_user_info.csv')))

    if dataset == 'ml-1m' or dataset == 'lfm-demobias':
        map = {'M': 1, 'F': 0} if dataset == 'ml-1m' else {'m': 1, 'f': 0}
        for fold in train_user_info:
            fold["gender"] = fold['gender'].map(map)
        for fold in val_user_info:
            fold["gender"] = fold['gender'].map(map)
        for fold in test_user_info:
            fold["gender"] = fold['gender'].map(map)
            
    return train_fold_list, val_fold_list, test_fold_list, train_user_info, val_user_info, test_user_info