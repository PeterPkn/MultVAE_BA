import argparse
import os

import pandas as pd
import torch
from NN_trainers.multVAE_trainer import train as train_multvae
from NN_trainers.multVAE_ADV_trainer import train as train_multvae_adv
from NN_trainers.multVAE_DA_trainer import train as train_multvae_da
from DataLoaders.ML1M_loader import get_dataset_dataloaders
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
    model_group.add_argument('--multvae_adv', action='store_true', help='Train the multVAE model with adversarial debiasing')
    model_group.add_argument('--multvae_da_visualize', action='store_true', help='Train the multVAE model with domain adaptation and visualize the bias by varying the CLUB weight')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--prioritize_bias', action='store_true', help='Prioritize bias reduction by selecting the model with the lowest balanced accuracy on the bias prediction task, instead of the best validation performance')
    parser.add_argument('--club_weight', type=float, default=7000.0, help='Weight for the CLUB penalty in the multVAE_DA model')
    parser.add_argument('--test_bias', action='store_true', help='Test the bias of the trained model using an adversarial network')
    parser.add_argument('--small_model', action='store_true', help='Use a smaller model of dimension: INPUT->500->200(latent)->OUTPUT.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and testing')
    parser.add_argument('--latent_dim_domain', type=int, default=200, help='Latent dimension for the domain encoder in the multVAE_DA model')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset to use (default: ml-1m), available options: ml-1m, ekstrabladed, lfm-demobias')
    parser.add_argument('--anneal_cap', type=float, default=0.4, help='Maximum weight for KL divergence annealing (default: 0.4)')
    parser.add_argument("--mi_estimator", type=str, default="L1Out", help='Mutual information estimator to use in CLUB penalty (default: L1Out), options: L1Out, CLUB, MINE, VUB')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--alpha', type=float, default=1.0, help='Dropout probability')
    parser.add_argument('--anneal_steps', type=int, default=45, help='Number of steps for KL divergence annealing in terms of epochs (default: 45)')
    # parse arguments
    args = parser.parse_args()

    # main logic here
    if args.multvae:
        train_loader, val_loader, test_loader, train_user_info, val_user_info, test_user_info = get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)
        for idx, fold in enumerate(train_loader):
            print("Fold ", idx)
            epochs = args.epochs
            train_multvae(epochs=epochs, train_user_info=train_user_info[idx], val_user_info=val_user_info[idx], test_user_info=test_user_info[idx], train_loader=DataLoader(fold, batch_size=args.batch_size, shuffle=True), val_loader=DataLoader(val_loader[idx], batch_size=args.batch_size, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=args.batch_size, shuffle=False), anneal_steps=len(fold)*args.anneal_steps, anneal_cap=args.anneal_cap, small_model=args.small_model, dropout=args.dropout, store_model=True)

    elif args.multvae_da:
        epochs = args.epochs
        train_loader, val_loader, test_loader, train_user_info, val_user_info, test_user_info = get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)
        results_bacc = []
        results_metrics = []
        for idx, fold in enumerate(train_loader):
            b_acc, metrics = train_multvae_da(epochs=epochs, train_loader=DataLoader(fold, batch_size=args.batch_size, shuffle=True), train_user_info=train_user_info[idx], val_user_info=val_user_info[idx], test_user_info=test_user_info[idx], val_loader=DataLoader(val_loader[idx], batch_size=args.batch_size, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=args.batch_size, shuffle=False), anneal_steps=len(fold)*args.anneal_steps, anneal_cap=args.anneal_cap, club_weight=args.club_weight, prioritize_bias=args.prioritize_bias, latent_dim_domain=args.latent_dim_domain, mi_estimator=args.mi_estimator, dropout=args.dropout)
            results_bacc.append(b_acc)
            results_metrics.append(metrics)
        print(f"Average Balanced Accuracy on Bias Prediction Task across folds: {np.mean(results_bacc):.4f}")
        print(f"Average NDCG@10 across folds: {np.mean([m['ndcg@10'] for m in results_metrics]):.4f}")
            
    elif args.multvae_adv:
        train_loader, val_loader, test_loader, train_user_info, val_user_info, test_user_info = get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)
        results_bacc = []
        results_metrics = []
        for idx, fold in enumerate(train_loader):
            print("Fold ", idx)
            epochs = args.epochs
            b_acc, metrics = train_multvae_adv(epochs=epochs, train_user_info=train_user_info[idx], val_user_info=val_user_info[idx], test_user_info=test_user_info[idx], train_loader=DataLoader(fold, batch_size=args.batch_size, shuffle=True), val_loader=DataLoader(val_loader[idx], batch_size=args.batch_size, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=args.batch_size, shuffle=False), anneal_steps=len(fold)*args.anneal_steps, anneal_cap=args.anneal_cap, small_model=args.small_model, alpha=args.alpha, adv_net_dim=100, dropout=args.dropout)
            results_bacc.append(b_acc)
            results_metrics.append(metrics)
        print(f"Average Balanced Accuracy on Bias Prediction Task across folds: {np.mean(results_bacc):.4f}")
        print(f"Average NDCG@10 across folds: {np.mean([m['ndcg@10'] for m in results_metrics]):.4f}")

    elif args.multvae_da_visualize:
        train_loader, val_loader, test_loader, train_user_info, val_user_info, test_user_info = get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)
        train_loader = train_loader[0]  # Just take the first fold for visualization
        val_loader = val_loader[0]
        test_loader = test_loader[0]
        train_user_info = train_user_info[0]
        club_weights = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        epochs = args.epochs
        all_results = {w: [] for w in club_weights}
        for club_weight in tqdm(club_weights):
            
            (b_acc, ndcg) = train_multvae_da(epochs=int(epochs), train_loader=DataLoader(train_loader, batch_size=args.batch_size, shuffle=True), train_user_info=train_user_info, val_user_info=val_user_info, test_user_info=test_user_info, val_loader=DataLoader(val_loader, batch_size=args.batch_size, shuffle=False), test_loader=DataLoader(test_loader, batch_size=args.batch_size, shuffle=False), anneal_steps=len(train_loader)*int(args.anneal_steps), anneal_cap=args.anneal_cap, club_weight=club_weight, prioritize_bias=args.prioritize_bias, latent_dim_domain=args.latent_dim_domain)
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
            plt.savefig('fairness_vs_utility_tradeoff_dimensiontest.png', dpi=300, bbox_inches='tight')
            plt.show()