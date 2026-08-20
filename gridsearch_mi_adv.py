import argparse
import itertools
from torch import optim
from NeuralNetworks.ADV_net import ADV_net
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import gc
import copy


# PIP install list: torch pandas numpy scipy scikit-learn rmet matplotlib tqdm

# Import your existing modules
from NN_trainers.multVAE_DA_trainer import train as train_multvae_da
from NN_trainers.multVAE_trainer import train as train_multvae
from DataLoaders.ML1M_loader import get_dataset_dataloaders

def run_grid_search(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    results = []
    print(f"Loading dataset: {args.dataset}")
    train_loader_list, val_loader_list, test_loader_list, train_info, val_info, test_info = get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)


    # Adversary grid search parameters
    param_grid = {
            'lr': [1e-3, 5e-4, 1e-4],
            'weight_decay': [0.0, 1e-4, 1e-5],
            'hidden_dim': [50, 100, 200],      
            'dropout': [0.0, 0.2, 0.5],        
            'adv_epochs': [50, 100]           
        }
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    

    for _, mi_est in enumerate(['CLUB', 'VUB', 'L1Out', 'MINE']):
        print(f"\n--- Running Function {mi_est} ---")

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
            b_acc, metrics, model = train_multvae_da(
                epochs=args.epochs,
                train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
                train_user_info=train_user_info,
                val_user_info=val_user_info,
                test_user_info=test_user_info,
                val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False),
                test_loader=DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
                anneal_steps=len(train_dataset) * args.anneal,
                anneal_cap=args.anneal_cap,
                club_weight=0.4,
                prioritize_bias=True,
                latent_dim_domain=200,
                mi_logvar=0.0,
                mi_estimator=mi_est
            )
            if metrics and 'ndcg@10' in metrics:
                avg_ndcg10 += float(metrics['ndcg@10']) / 5.0  # Average NDCG@10 over 5 folds

            train_dataset = DataLoader(train_loader_list[fold], batch_size=args.batch_size, shuffle=True)
            val_dataset = DataLoader(val_loader_list[fold], batch_size=args.batch_size, shuffle=False)
            test_dataset = DataLoader(test_loader_list[fold], batch_size=args.batch_size, shuffle=False)
            
            # Colletc data for adv training,
            features_for_pca = []
            lables_for_visual = []
            model.eval()
            for _, (x_data, _, idx) in enumerate(train_dataset):
                    x_data = torch.FloatTensor(x_data).to(device)
                    mu = None
                    if args.model_type == 'multvae':
                        _, mu, _ = model(x_data)
                    elif args.model_type == 'multvae_da':
                        _, mu, _, _, _, _ = model(x_data)
                    elif args.model_type == 'multvae_adv':
                        _, mu, _, _ = model(x_data)
                    if mu is None:
                        raise ValueError(f"Model did not return mu for model_type: {args.model_type}")
                    features_for_pca.extend(mu.cpu().detach().numpy().tolist())
                    lables_for_visual.extend(train_user_info.iloc[idx.numpy()]['gender'].tolist())

            num_men = sum(lables_for_visual)
            features_len = len(lables_for_visual)

            features_for_pca = np.array(features_for_pca, dtype=np.float32)
            lables_for_visual = np.array(lables_for_visual, dtype=np.float64)

            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import balanced_accuracy_score

            # Standardize the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(features_for_pca)

            val_features_for_pca = []
            val_lables_for_visual = []
            for _, (x_data, _, idx) in enumerate(val_dataset):
                    x_data = torch.FloatTensor(x_data).to(device)
                    mu = None
                    if args.model_type == 'multvae':
                        _, mu, _ = model(x_data)
                    elif args.model_type == 'multvae_da':
                        _, mu, _, _, _, _ = model(x_data)
                    elif args.model_type == 'multvae_adv':
                        _, mu, _, _ = model(x_data)
                    if mu is None:
                        raise ValueError(f"Model did not return mu for model_type: {args.model_type}")
                    val_features_for_pca.extend(mu.cpu().detach().numpy().tolist())
                    val_lables_for_visual.extend(val_user_info.iloc[idx.numpy()]['gender'].tolist())

            val_features_for_pca = np.array(val_features_for_pca, dtype=np.float32)
            val_lables_for_visual = np.array(val_lables_for_visual, dtype=np.float64)

            X_val_scaled = scaler.transform(val_features_for_pca)


            max_bacc = 0.0
            best_model = None
            best_adv_dropout = None
            best_adv_hidden = None
            # Train ADV on Train -> evaulate performance on Validation and then at the end check B_ACC on Test set
            for i, combo in enumerate(combinations):
                params = dict(zip(keys, combo))
                print(f"\n--- Running Combination {i+1}/{len(combinations)} ---")
                print(f"Parameters: {params}")
    
                
                # TRAIN ADV-net
                adv_model = ADV_net(200, params['hidden_dim'], dropout=params['dropout'])
                adv_optim = optim.Adam(
                    adv_model.parameters(),
                    lr=params['lr'],
                    weight_decay=params['weight_decay']
                )
    
                weights = torch.tensor([features_len/(2*(features_len-num_men)),features_len/(2*num_men)], dtype=torch.float32)
                weights = weights.to(device)
                adv_loss = torch.nn.CrossEntropyLoss(weights)

                adv_model.to(device)
                adv_model.train()
                for _ in range(params['adv_epochs']):
                    permuted_indices = np.random.permutation(X_train_scaled.shape[0])
                    training_samples = X_train_scaled[permuted_indices]
                    training_labels = lables_for_visual[permuted_indices]
                    for batch in range(np.ceil(X_train_scaled.shape[0]/args.batch_size).astype(int)):
                        
                        x_data = torch.from_numpy(training_samples[batch*args.batch_size:np.min([batch*args.batch_size+args.batch_size, training_samples.shape[0]])])
                        y_data = torch.from_numpy(training_labels[batch*args.batch_size:np.min([batch*args.batch_size+args.batch_size, training_labels.shape[0]])]).long()
    
                        x_data = x_data.to(device)
                        y_data = y_data.to(device)
    
                        adv_optim.zero_grad()
                        predictions = adv_model(x_data)
                        loss = adv_loss(predictions, y_data)
    
                        loss.backward()
                        adv_optim.step()
                adv_b_acc = 0.0
    
                adv_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.from_numpy(X_val_scaled).to(device)
                    predictions_val = adv_model(X_val_tensor)
    
                    all_preds = torch.argmax(predictions_val, dim=1).cpu().numpy()
                    all_true = val_lables_for_visual
    
                    adv_b_acc = balanced_accuracy_score(all_true, all_preds)
                    #print(f"Standardized Balanced Accuracy: {b_acc}")

                if adv_b_acc >= max_bacc:
                    max_bacc = adv_b_acc
                    best_model_state = copy.deepcopy(adv_model.state_dict())
                    best_adv_hidden = params['hidden_dim']
                    best_adv_dropout = params['dropout']

                del adv_optim, adv_loss
                del predictions_val, all_preds, all_true

            #check the accuracy on the test data, has not seen this data befor -> cannot overfit  
            test_features_for_pca = []
            test_lables_for_visual = []
            for _, (x_data, _, idx) in enumerate(test_dataset):
                    x_data = torch.FloatTensor(x_data).to(device)
                    mu = None
                    if args.model_type == 'multvae':
                        _, mu, _ = model(x_data)
                    elif args.model_type == 'multvae_da':
                        _, mu, _, _, _, _ = model(x_data)
                    elif args.model_type == 'multvae_adv':
                        _, mu, _, _ = model(x_data)
                    if mu is None:
                        raise ValueError(f"Model did not return mu for model_type: {args.model_type}")
                    test_features_for_pca.extend(mu.cpu().detach().numpy().tolist())
                    test_lables_for_visual.extend(test_user_info.iloc[idx.numpy()]['gender'].tolist())

            test_features_for_pca = np.array(test_features_for_pca, dtype=np.float32)
            test_lables_for_visual = np.array(test_lables_for_visual, dtype=np.float64)

            X_test_scaled = scaler.transform(test_features_for_pca)
            test_b_acc = 0.0
            if best_model is not None and best_adv_hidden is not None and best_adv_dropout is not None:
                final_adv = ADV_net(200, best_adv_hidden, dropout=best_adv_dropout).to(device)
                final_adv.load_state_dict(best_model)
                final_adv.eval()
                with torch.no_grad():
                    X_test_scaled = torch.from_numpy(X_test_scaled).to(device)
                    predictions_test = best_model(X_test_scaled)

                    all_preds = torch.argmax(predictions_test, dim=1).cpu().numpy()
                    all_true = test_lables_for_visual

                    test_b_acc = balanced_accuracy_score(all_true, all_preds)
                    #print(f"Standardized Balanced Accuracy: {b_acc}")

            del train_dataset, val_dataset, test_dataset
            del train_user_info, val_user_info, test_user_info

            gc.collect()

            torch.cuda.empty_cache()

            avg_b_acc += float(test_b_acc) / 5.0
            



        result_entry = {
            'mi_est': mi_est,
            'balanced_accuracy': avg_b_acc,
            'ndcg@10': avg_ndcg10,
        }
            
        results.append(result_entry)

        df_entry = pd.DataFrame([result_entry])
        file_exists = os.path.isfile(f"progress_{args.output_file}")
        df_entry.to_csv(f"progress_{args.output_file}", index=False, mode="a", header=not file_exists)

    # save Results to CSV
    df_results = pd.DataFrame(results)
    

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
    parser.add_argument('--anneal', type=int, default=45, help='KLD anneal period in epochs')
    parser.add_argument('--anneal_cap', type=float, default=0.1, help='KLD anneal cap')
    parser.add_argument('--output_file', type=str, default='grid_search_results.csv', help='CSV file to save results')
    parser.add_argument('--model_type', type=str, choices=['multvae', 'multvae_da', 'multvae_adv'], default='multvae_da', help='Type of model to train (default: multvae_da)', required=True)

    
    args = parser.parse_args()
    
    run_grid_search(args)