import argparse
from NN_trainers.multVAE_trainer import train
from DataLoaders.ML1M_loader import get_ml1m_dataloaders
from torch.utils.data import DataLoader

train_loader, val_loader, test_loader = get_ml1m_dataloaders()
for idx, fold in enumerate(train_loader):
    print("Fold ", idx)
    epochs = 70
    loader = DataLoader(fold, batch_size=128, shuffle=True)
    batch = next(iter(loader))
    print(batch[0][0])