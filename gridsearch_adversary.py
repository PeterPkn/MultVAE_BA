import argparse
import itertools
from xml.parsers.expat import model
import pandas as pd
from NeuralNetworks.ADV_net import ADV_net
from NeuralNetworks.multVAE import MultVAE
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import os
import gc
from tqdm import tqdm
import os
import gc
import time

# Import your existing modules
from NN_trainers.multVAE_DA_trainer import train as train_multvae_da
from NN_trainers.multVAE_trainer import train as train_multvae
from DataLoaders.ML1M_loader import get_dataset_dataloaders

def run_grid_search(args):
    param_grid = {
        'lr': [1e-3, 5e-4, 1e-4],
        'weight_decay': [0.0, 1e-4, 1e-5],
        'hidden_dim': [50, 100, 200],      
        'dropout': [0.0, 0.2, 0.5],        
        'adv_epochs': [50, 100]           
    }

    train_loader, _, _, train_user_info, val_user_info, test_user_info = get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)
        

    standard_model = [3416, 600, 200]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MultVAE(standard_model, latent_dim=200, dropout=0.6, training=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    
    print(f"Starting Grid Search with {len(combinations)} combinations...")
    
    print(f"Loading dataset: {args.dataset}")
    train_loader_list, val_loader_list, test_loader_list, train_info, val_info, test_info = \
        get_dataset_dataloaders(global_indexing=False, dataset=args.dataset)

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n--- Running Combination {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")

        avg_b_acc = 0.0

        for fold in range(5):
            print(f"\n--- Fold {fold+1}/5 ---")
            if args.skip_epoch > 0 and fold + 1 == args.skip_epoch:
                print(f"Skipping fold {fold+1} as per --skip_epoch argument.")
                continue
            train_dataset = DataLoader(train_loader_list[fold], batch_size=args.batch_size, shuffle=True)
            val_dataset = DataLoader(val_loader_list[fold], batch_size=args.batch_size, shuffle=False)
            test_dataset = DataLoader(test_loader_list[fold], batch_size=args.batch_size, shuffle=False)

            train_user_info = train_info[fold]
            val_user_info = val_info[fold]
            test_user_info = test_info[fold]

            features_for_pca = []
            lables_for_visual = []
            model.eval()
            for _, (x_data, _, idx) in enumerate(train_dataset):
                    x_data = torch.FloatTensor(x_data).to(device)
                    _, mu, logvar = model(x_data)
                    features_for_pca.extend(mu.cpu().detach().numpy().tolist())
                    lables_for_visual.extend(train_user_info.iloc[idx.numpy()]['gender'].tolist())


            num_men = sum(lables_for_visual)
            features_len = len(lables_for_visual)

            features_for_pca = np.array(features_for_pca, dtype=np.float32)
            lables_for_visual = np.array(lables_for_visual, dtype=np.float64)

            
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

            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import balanced_accuracy_score

            # Standardize the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(features_for_pca)

            adv_model.to(device)
            adv_model.train()
            for _ in tqdm(range(params['adv_epochs'])):
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
            b_acc = 0.0

            features_for_pca = []
            lables_for_visual = []
            for _, (x_data, _, idx) in enumerate(test_dataset):
                    x_data = torch.FloatTensor(x_data).to(device)
                    _, mu, logvar = model(x_data)
                    features_for_pca.extend(mu.cpu().detach().numpy().tolist())
                    lables_for_visual.extend(test_user_info.iloc[idx.numpy()]['gender'].tolist())

            features_for_pca = np.array(features_for_pca, dtype=np.float32)
            lables_for_visual = np.array(lables_for_visual, dtype=np.float64)

            X_test_scaled = scaler.transform(features_for_pca)

            adv_model.eval()
            with torch.no_grad():
                X_test_scaled = torch.from_numpy(X_test_scaled).to(device)
                predictions_test = adv_model(X_test_scaled)

                all_preds = torch.argmax(predictions_test, dim=1).cpu().numpy()
                all_true = lables_for_visual

                b_acc = balanced_accuracy_score(all_true, all_preds)
                #print(f"Standardized Balanced Accuracy: {b_acc}")

            
            avg_b_acc += float(b_acc) / (5.0 if args.skip_epoch == 0 else 4.0)  # Average over 5 folds, adjusting for skipped fold

            del train_dataset, val_dataset, test_dataset
            del train_user_info, val_user_info, test_user_info

            gc.collect()

            torch.cuda.empty_cache()


        result_entry = {
            **params,
            'balanced_accuracy': avg_b_acc,
        }
            
        results.append(result_entry)

        df_entry = pd.DataFrame([result_entry])
        file_exists = os.path.isfile(f"progress_{args.output_file}")
        df_entry.to_csv(f"progress_{args.output_file}", index=False, mode="a", header=not file_exists)

    # save Results to CSV
    df_results = pd.DataFrame(results)
    
    # sort the dataframe so the best NDCG@10 models are at the top
    if 'ndcg@10' in df_results.columns:
        df_results = df_results.sort_values(by=['ndcg@10'], ascending=False)
        
    df_results.to_csv(f"final_{args.output_file}", index=False)
    print(f"\nGrid search complete! Results saved to {args.output_file}")
    
    print("\nTop 3 Configurations by NDCG@10 (if available):")
    print(df_results.head(3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for multVAE_DA model")
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset to use (default: ml-1m)')
    #parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs per combination')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_file', type=str, default='grid_search_results.csv', help='CSV file to save results')
    parser.add_argument('--model_path', type=str, default='./ml1m_multvae.pth', help='Path to the pre-trained multVAE model')
    parser.add_argument('--skip_epoch', type=int, default=0, help='Skip the epoch the MultVAE model was trained on (default: 0, meaning no skip, 1-5 for skipping the corresponding epoch)')
    args = parser.parse_args()
    run_grid_search(args)