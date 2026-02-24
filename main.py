import argparse
from NN_trainers.multVAE_trainer import train as train_multvae
from NN_trainers.multVAE_DA_trainer import train as train_multvae_da
from DataLoaders.ML1M_loader import get_ml1m_dataloaders
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

# console app boilerplate
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multVAE main script")
    # add arguments here
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--multvae', action='store_true', help='Train the multVAE model')
    model_group.add_argument('--multvae_da', action='store_true', help='Train the multVAE model with domain adaptation')
    model_group.add_argument('--multvae_da_visualize', action='store_true', help='Train the multVAE model with domain adaptation and visualize the bias by varying the CLUB weight')

    # parse arguments
    args = parser.parse_args()

    # main logic here
    if args.multvae:
        train_loader, val_loader, test_loader, train_user_info = get_ml1m_dataloaders()
        for idx, fold in enumerate(train_loader):
            print("Fold ", idx)
            epochs = 70
            train_multvae(epochs=epochs, train_loader=DataLoader(fold, batch_size=128, shuffle=True), val_loader=DataLoader(val_loader[idx], batch_size=128, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=128, shuffle=False), anneal_steps=len(fold)*45, anneal_cap=0.2)

    elif args.multvae_da:
        train_loader, val_loader, test_loader, train_user_info = get_ml1m_dataloaders()
        for idx, fold in enumerate(train_loader):
            print("Fold ", idx)
            epochs = 70
            train_multvae_da(epochs=epochs, train_loader=DataLoader(fold, batch_size=128, shuffle=True), train_user_info=train_user_info, val_loader=DataLoader(val_loader[idx], batch_size=128, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=128, shuffle=False), anneal_steps=len(fold)*45, anneal_cap=0.2)
    elif args.multvae_da_visualize:
        train_loader, val_loader, test_loader, train_user_info = get_ml1m_dataloaders()
        club_weights = [0.0, 1000.0, 3000.0, 5000.0, 7000.0, 9000.0, 15000.0]
        epochs = 1
        all_results = {w: [] for w in club_weights}

        for idx, fold in enumerate(train_loader):
            print("Fold ", idx)
            for club_weight in club_weights:
                (b_acc, ndcg) = train_multvae_da(epochs=int(epochs+(club_weight/1000)), train_loader=DataLoader(fold, batch_size=128, shuffle=True), train_user_info=train_user_info, val_loader=DataLoader(val_loader[idx], batch_size=128, shuffle=False), test_loader=DataLoader(test_loader[idx], batch_size=128, shuffle=False), anneal_steps=len(fold)*int(45+(club_weight/1000)), anneal_cap=0.2, club_weight=club_weight)
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