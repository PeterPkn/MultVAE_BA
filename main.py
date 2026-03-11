import argparse
import os

import pandas as pd
import torch
from NN_trainers.multVAE_trainer import train as train_multvae
from NN_trainers.multVAE_DA_trainer import train as train_multvae_da
from DataLoaders.ML1M_loader import get_ml1m_dataloaders
from NN_testing.test_bias import test
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

# console app boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multVAE main script")
    # add arguments here
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--multvae', action='store_true', help='Train the multVAE model')
    model_group.add_argument('--multvae_da', action='store_true', help='Train the multVAE model with domain adaptation')
    model_group.add_argument('--multvae_da_visualize', action='store_true', help='Train the multVAE model with domain adaptation and visualize the bias by varying the CLUB weight')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--alldata', action='store_true', help='Use all data for training (no validation set)')
    parser.add_argument('--prioritize_bias', action='store_true', help='Prioritize bias reduction by selecting the model with the lowest balanced accuracy on the bias prediction task, instead of the best validation performance')
    parser.add_argument('--club_weight', type=float, default=7000.0, help='Weight for the CLUB penalty in the multVAE_DA model')
    parser.add_argument('--test_bias', action='store_true', help='Test the bias of the trained model using an adversarial network')
    parser.add_argument('--small_model', action='store_true', help='Use a smaller model of dimension: INPUT->500->200(latent)->OUTPUT.')

    # parse arguments
    args = parser.parse_args()

    # main logic here
    if args.multvae:
        train_loader, val_loader, test_loader, train_user_info = get_ml1m_dataloaders()

        if args.alldata:
            train_loader = torch.utils.data.ConcatDataset(train_loader)
            val_loader = torch.utils.data.ConcatDataset(val_loader)
            test_loader = torch.utils.data.ConcatDataset(test_loader)
            train_user_info = pd.concat(train_user_info, ignore_index=True)

            if args.test_bias:
                test(train_loader=DataLoader(train_loader, batch_size=128, shuffle=False), train_user_info=train_user_info, test_loader=DataLoader(test_loader, batch_size=128, shuffle=False), val_loader=DataLoader(val_loader, batch_size=128, shuffle=False), small_model=args.small_model)
            else:
                train_multvae(epochs=args.epochs, train_loader=DataLoader(train_loader, batch_size=128, shuffle=True), val_loader=DataLoader(val_loader, batch_size=128, shuffle=False), test_loader=DataLoader(test_loader, batch_size=128, shuffle=False), train_user_info=train_user_info, anneal_steps=len(train_loader)*45, anneal_cap=0.2, small_model=args.small_model)

        else:
            for idx, fold in enumerate(train_loader):
                print("Fold ", idx)
                epochs = args.epochs
                train_multvae(epochs=epochs, train_loader=DataLoader(fold, batch_size=128, shuffle=True), val_loader=DataLoader(val_loader[idx], batch_size=128, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=128, shuffle=False), anneal_steps=len(fold)*45, anneal_cap=0.2, small_model=args.small_model)

    elif args.multvae_da:
        train_loader, val_loader, test_loader, train_user_info = get_ml1m_dataloaders()
        epochs = args.epochs
        if args.alldata:
            train_loader = torch.utils.data.ConcatDataset(train_loader)
            val_loader = torch.utils.data.ConcatDataset(val_loader)
            test_loader = torch.utils.data.ConcatDataset(test_loader)
            train_user_info = pd.concat(train_user_info, ignore_index=True)
            #print(len(train_loader))
            train_multvae_da(epochs=epochs, train_loader=DataLoader(train_loader, batch_size=128, shuffle=True), train_user_info=train_user_info, val_loader=DataLoader(val_loader, batch_size=128, shuffle=False), test_loader=DataLoader(test_loader, batch_size=128, shuffle=False), anneal_steps=len(train_loader)*45, anneal_cap=0.2, club_weight=args.club_weight, prioritize_bias=args.prioritize_bias)
            
        else:
            for idx, fold in enumerate(train_loader):
                print("Fold ", idx)
                train_multvae_da(epochs=epochs, train_loader=DataLoader(fold, batch_size=128, shuffle=True), train_user_info=train_user_info[idx], val_loader=DataLoader(val_loader[idx], batch_size=128, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=128, shuffle=False), anneal_steps=len(fold)*120, anneal_cap=0.2, club_weight=args.club_weight, prioritize_bias=args.prioritize_bias)
    elif args.multvae_da_visualize:
        train_loader, val_loader, test_loader, train_user_info = get_ml1m_dataloaders()
        train_loader = torch.utils.data.ConcatDataset(train_loader)
        val_loader = torch.utils.data.ConcatDataset(val_loader)
        test_loader = torch.utils.data.ConcatDataset(test_loader)
        train_user_info = pd.concat(train_user_info, ignore_index=True)
        club_weights = [0.1, 0.5, 1, 5, 10]
        epochs = args.epochs
        all_results = {w: [] for w in club_weights}
        for club_weight in tqdm(club_weights):
            
            (b_acc, ndcg) = train_multvae_da(epochs=int(epochs), train_loader=DataLoader(train_loader, batch_size=128, shuffle=True), train_user_info=train_user_info, val_loader=DataLoader(val_loader, batch_size=128, shuffle=False), test_loader=DataLoader(test_loader, batch_size=128, shuffle=False), anneal_steps=len(train_loader)*int(45), anneal_cap=0.2, club_weight=club_weight, prioritize_bias=args.prioritize_bias)
            print(f"Club weight: {club_weight}, Best accuracy: {b_acc}, NDCG: {ndcg}")
            all_results[club_weight].append((b_acc, ndcg))

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

        # Dynamically check how many folds completed
        num_completed_folds = len(all_results[club_weights[0]])
        valid_weights = [w for w in club_weights if len(all_results[w]) == num_completed_folds]

        if num_completed_folds > 0:
            # Extract averages across the folds
            avg_accs = [np.mean([all_results[w][i][0] for i in range(num_completed_folds)]) for w in valid_weights]
            avg_ndcgs = [np.mean([all_results[w][i][1] for i in range(num_completed_folds)]) for w in valid_weights]

            # 1. Plot individual folds (lighter, dashed lines)
            for fold_idx in range(num_completed_folds):
                fold_accs = [all_results[w][fold_idx][0] for w in valid_weights]
                fold_ndcgs = [all_results[w][fold_idx][1] for w in valid_weights]
                
                # Blue for Bias, Green for NDCG
                ax1.plot(valid_weights, fold_accs, marker='o', linestyle='--', color='blue', alpha=0.15)
                ax2.plot(valid_weights, fold_ndcgs, marker='^', linestyle='--', color='green', alpha=0.15)

            # 2. Plot Averages (bold, solid lines)
            line1 = ax1.plot(valid_weights, avg_accs, marker='s', color='blue', linewidth=2.5, label='Avg Bias (Balanced Acc)')
            line2 = ax2.plot(valid_weights, avg_ndcgs, marker='D', color='green', linewidth=2.5, label='Avg Utility (NDCG@10)')

            # 3. Add a baseline for "Perfect Fairness" on ax1
            line3 = ax1.axhline(y=0.5, color='red', linestyle=':', linewidth=2, label='Random Guess (0.5 Bias)')

            # 4. Formatting for thesis-level quality
            plt.title('Fairness vs. Utility Tradeoff: Effect of MI Penalty', fontsize=14, fontweight='bold')
            ax1.set_xlabel('CLUB Weight (MI Penalty)', fontsize=12)
            
            # Format left axis (Bias)
            ax1.set_ylabel('ADV_Net Balanced Accuracy (Bias)', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(0.45, 0.85) 
            
            # Format right axis (Utility)
            ax2.set_ylabel('Recommendation Performance (NDCG@10)', fontsize=12, color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            # ax2.set_ylim(0.20, 0.40) # Optional: uncomment and adjust if you want to lock the NDCG scale

            ax1.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Combine legends from both axes into one box
            lines = line1 + line2 + [line3]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper center')

            # Save the figure
            plt.savefig('fairness_vs_utility_tradeoff.png', dpi=300, bbox_inches='tight')
            plt.show()