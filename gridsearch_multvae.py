import argparse
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import gc
import time


# PIP install list: torch pandas numpy scipy scikit-learn rmet matplotlib tqdm

# Import your existing modules
from NN_trainers.multVAE_DA_trainer import train as train_multvae_da
from NN_trainers.multVAE_trainer import train as train_multvae
from NN_trainers.multVAE_ADV_trainer import train as train_multvae_adv
from DataLoaders.ML1M_loader import get_dataset_dataloaders

def run_grid_search(args):
    _param_grid = {
    # 'mi_logvar': [-1, -1.5],
    # 'club_weight': [100.0, 200.0, 500.0], 
    # 'latent_dim_domain': [50],
    # 'prioritize_bias': [True],
    # 'anneal_cap': [0.1],
    # 'anneal_period': [75],
    # 'batch_size': [1024],
    # 'epochs': [150],
    # 'dropout': [0.6]
    }
    if args.model_type == 'multvae_da':

        param_grid = {
        'estimator': ['VUB' ,'CLUB', 'L1Out', 'MINE'],
        'club_weight': [0.008, 0.4],
        'mi_logvar': [0.0],
        'latent_dim_domain': [200],
        'prioritize_bias': [True],
        'anneal_cap': [0.1],
        'anneal_period': [45],
        'batch_size': [1024],
        'epochs': [100],
        'dropout': [0.6],
        'use_bound': [False]
        }

        if args.dataset == 'lfm-demobias':
            param_grid = {
                'estimator': ['VUB' ,'CLUB', 'L1Out', 'MINE'],
                'club_weight': [0.2],
                'mi_logvar': [0.0],
                'latent_dim_domain': [200],
                'prioritize_bias': [True],
                'anneal_cap': [0.1],
                'anneal_period': [60],
                'batch_size': [1024],
                'epochs': [50],
                'dropout': [0.4],
                'use_bound': [False]
                }
    elif args.model_type == 'multvae':
        param_grid = {
            'anneal_cap': [0.1],
            'anneal_period': [45],
            'batch_size': [1024],
            'epochs': [100],
            'dropout': [0.6]
        }
        if args.dataset == 'lfm-demobias':
            param_grid = {
                'anneal_cap': [0.1],
                'anneal_period': [60],
                'batch_size': [1024],
                'epochs': [50],
                'dropout': [0.4]
            }
    else:# for multvae_adv
        param_grid = {
            'adv_net_dim': [100, 50, 200],
            'alpha': [100.0, 500.0, 1000.0, 2300.0, 3000.0],
            'anneal_cap': [0.1],
            'anneal_period': [45],
            'batch_size': [1024],
            'epochs': [100],
            'dropout': [0.6]
        }
        if args.dataset == 'lfm-demobias':
            param_grid = {
                'adv_net_dim': [100, 50, 200],
                'alpha': [100.0, 500.0, 1000.0, 2300.0, 3000.0],
                'anneal_cap': [0.1],
                'anneal_period': [60],
                'batch_size': [1024],
                'epochs': [50],
                'dropout': [0.4]
            }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    
    print(f"Starting Grid Search with {len(combinations)} combinations...")
    
    print(f"Loading dataset: {args.dataset}")
    train_loader_list, val_loader_list, test_loader_list, train_info, val_info, test_info = \
        get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)

    results = []

    progress_file = f"progress_{args.output_file}"
    completed_combos = set()

    if os.path.isfile(progress_file):
        print(f"Found existing progress file: {progress_file}. Resuming...")
        df_prog = pd.read_csv(progress_file)
        
        for _, row in df_prog.iterrows():
            try:
                combo = tuple(type(param_grid[k][0])(row[k]) for k in keys)
                completed_combos.add(combo)
            except ValueError:
                pass
                
        print(f"Loaded {len(completed_combos)} previously completed combinations.")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Running Combination {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        if combo in completed_combos:
            print(f"Skipping Combination {i+1}/{len(combinations)} (Already completed)")
            continue

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
            if args.model_type == 'multvae':
                b_acc, metrics, _ = train_multvae(
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
            elif args.model_type == 'multvae_da':
                b_acc, metrics, _ = train_multvae_da(
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
                    mi_logvar=params['mi_logvar'],
                    mi_estimator=params['estimator'],
                    use_bound=params['use_bound'],
                    dropout=params['dropout']
                )
            else:
                b_acc, metrics, _ = train_multvae_adv(
                    epochs=params['epochs'],
                    train_loader=DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True),
                    train_user_info=train_user_info,
                    val_user_info=val_user_info,
                    test_user_info=test_user_info,
                    val_loader=DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False),
                    test_loader=DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False),
                    anneal_steps=len(train_dataset) * params['anneal_period'],
                    anneal_cap=params['anneal_cap'],
                    adv_net_dim=params['adv_net_dim'],
                    alpha=params['alpha'],
                    store_info=True,
                    dropout=params['dropout']
                )                
            avg_b_acc += float(b_acc) / 5.0  # Average over 5 folds
            if metrics and 'ndcg@10' in metrics:
                avg_ndcg10 += float(metrics['ndcg@10']) / 5.0  # Average NDCG@10 over 5 folds

            del train_dataset, val_dataset, test_dataset
            del train_user_info, val_user_info, test_user_info

            gc.collect()

            torch.cuda.empty_cache()

            time.sleep(10)


        result_entry = {
            **params,
            'balanced_accuracy': avg_b_acc,
            'ndcg@10': avg_ndcg10,
        }
            
        results.append(result_entry)

        df_entry = pd.DataFrame([result_entry])
        file_exists = os.path.isfile(f"progress_{args.output_file}")
        df_entry.to_csv(f"progress_{args.output_file}", index=False, mode="a", header=not file_exists)

    # save Results to CSV
    df_results = pd.DataFrame(results)
    
    # sort the dataframe so the best NDCG@10 models are at the top
    if args.model_type == 'multvae':
        if 'ndcg@10' in df_results.columns:
            df_results = df_results.sort_values(by=['ndcg@10'], ascending=False)
    else:
        if 'balanced_accuracy' in df_results.columns:
            df_results = df_results.sort_values(by=['balanced_accuracy'], ascending=False)
            
    df_results.to_csv(f"final_{args.output_file}", index=False)
    print(f"\nGrid search complete! Results saved to {args.output_file}")
    
    print("\nTop 3 Configurations by NDCG@10 (if available):")
    print(df_results.head(3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for multVAE_DA model")
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset to use (default: ml-1m)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs per combination')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_file', type=str, default='grid_search_results.csv', help='CSV file to save results')
    parser.add_argument('--model_type', type=str, choices=['multvae', 'multvae_da', 'multvae_adv'], default='multvae_da', help='Type of model to train (default: multvae_da)', required=True)
    
    args = parser.parse_args()
    
    run_grid_search(args)
