import argparse
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os

# Import your existing modules
from NN_trainers.multVAE_DA_trainer import train as train_multvae_da
from NN_trainers.multVAE_trainer import train as train_multvae
from DataLoaders.ML1M_loader import get_dataset_dataloaders

def run_grid_search(args):
    # 1. Define the Hyperparameter Grid
    # You can easily add or modify parameters here
    param_grid = {
    'mi_logvar': [-1, -1.5],
    'club_weight': [100.0, 200.0, 500.0], 
    'latent_dim_domain': [50],
    'prioritize_bias': [True],
    'anneal_cap': [0.1],
    'anneal_period': [75],
    'batch_size': [1024],
    'epochs': [150],
    'dropout': [0.6]
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    
    print(f"Starting Grid Search with {len(combinations)} combinations...")
    
    # 2. Load the Dataset
    # We use global_indexing=False and grab the first fold for the grid search
    print(f"Loading dataset: {args.dataset}")
    train_loader_list, val_loader_list, test_loader_list, train_info, val_info, test_info = \
        get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)

    results = []

    # 3. Iterate through all combinations
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Running Combination {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")

        avg_b_acc = 0.0
        avg_ndcg10 = 0.0

        for fold in range(5):
            print(f"\n--- Fold {fold+1}/5 ---")
            train_dataset = train_loader_list[fold]
            val_dataset = val_loader_list[fold]
            test_dataset = test_loader_list[fold]
            
            train_user_info = train_info[fold]
            val_user_info = val_info[fold]
            test_user_info = test_info[fold]
            # Execute training process with the current parameters
            if args.noDA:
                b_acc, metrics = train_multvae(
                    epochs=params['epochs'],
                    train_loader=DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True),
                    train_user_info=train_user_info,
                    val_user_info=val_user_info,
                    test_user_info=test_user_info,
                    val_loader=DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False),
                    test_loader=DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False),
                    anneal_steps=len(train_dataset) * params['anneal_period'],
                    anneal_cap=params['anneal_cap'],
                    dropout=params['dropout']
                )
            else:
                b_acc, metrics = train_multvae_da(
                    epochs=params['epochs'],
                    train_loader=DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True),
                    train_user_info=train_user_info,
                    val_user_info=val_user_info,
                    test_user_info=test_user_info,
                    val_loader=DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False),
                    test_loader=DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False),
                    anneal_steps=len(train_dataset) * params['anneal_period'],
                    anneal_cap=params['anneal_cap'],
                    club_weight=params['club_weight'],
                    prioritize_bias=params['prioritize_bias'],
                    latent_dim_domain=params['latent_dim_domain'],
                    mi_logvar=params['mi_logvar']
                )
            avg_b_acc += b_acc / 5.0  # Average over 5 folds
            if metrics and 'ndcg@10' in metrics:
                avg_ndcg10 += metrics['ndcg@10'] / 5.0  # Average NDCG@10 over 5 folds


        # 4. Record Results
        result_entry = {
            **params,
            'balanced_accuracy': avg_b_acc,
            'ndcg@10': avg_ndcg10,
        }
            
        results.append(result_entry)

        df_entry = pd.DataFrame(result_entry)
        df_entry.to_csv(f"progress_{args.output_file}", index=False, mode="a")

    # 5. Save Results to CSV
    df_results = pd.DataFrame(results)
    
    # Sort the dataframe so the best NDCG@10 models are at the top
    if 'ndcg@10' in df_results.columns:
        df_results = df_results.sort_values(by=['ndcg@10'], ascending=False)
        
    df_results.to_csv(f"final_{args.output_file}", index=False)
    print(f"\nGrid search complete! Results saved to {args.output_file}")
    
    # Print the top 3 configurations
    print("\nTop 3 Configurations by NDCG@10 (if available):")
    print(df_results.head(3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for multVAE_DA model")
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset to use (default: ml-1m)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs per combination')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_file', type=str, default='grid_search_results.csv', help='CSV file to save results')
    parser.add_argument('--noDA', action='store_true', help='Gridsearch a pure MultVAE')
    
    args = parser.parse_args()
    
    run_grid_search(args)