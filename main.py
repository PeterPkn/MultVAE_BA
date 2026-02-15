import argparse
from NN_trainers.multVAE_trainer import train
from DataLoaders.ML1M_loader import get_ml1m_dataloaders
from torch.utils.data import DataLoader

# console app boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multVAE main script")
    # add arguments here
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--multvae', action='store_true', help='Train the multVAE model')
    model_group.add_argument('--multvae_da', action='store_true', help='Train the multVAE model with domain adaptation')

    # parse arguments
    args = parser.parse_args()

    # main logic here
    if args.multvae:
        train_loader, val_loader, test_loader = get_ml1m_dataloaders()
        for idx, fold in enumerate(train_loader):
            print("Fold ", idx)
            epochs = 70
            train(epochs=epochs, train_loader=DataLoader(fold, batch_size=128, shuffle=True), val_loader=DataLoader(val_loader[idx], batch_size=128, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=128, shuffle=False), anneal_steps=len(fold)*45, anneal_cap=0.2)

    elif args.multvae_da:
        train_loader, val_loader, test_loader = get_ml1m_dataloaders()
        for idx, fold in enumerate(train_loader):
            print("Fold ", idx)
            train(epochs=30, train_loader=DataLoader(fold, batch_size=128, shuffle=True), val_loader=DataLoader(val_loader[idx], batch_size=128, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=128, shuffle=False), anneal_steps=10000, anneal_cap=0.8)