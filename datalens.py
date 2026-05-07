import argparse
from NN_trainers.multVAE_trainer import train
from DataLoaders.ML1M_loader import get_dataset_dataloaders
from torch.utils.data import DataLoader

train_loader, val_loader, test_loader, _, _, _ = get_dataset_dataloaders(dataset="lfm-demobias")
for idx, fold in enumerate(train_loader):
    print("Fold ", idx)
    print(sum([elem_x.sum() for idx, (elem_x, elem_y, _) in enumerate(fold)])/len(fold))

# Max values for recall:
#   ekstrabladet: 0.034
#   lfm-demobias: 0.071
#   ml-1m: 0.06