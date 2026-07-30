from ast import mod
import datetime
import os
import random
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from rmet import BatchEvaluator
import numpy as np
from tqdm import tqdm
from NeuralNetworks.ADV_net import ADV_net
from NeuralNetworks.multVAE_ADV import MultVAE_ADV
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import datetime

def save_training_log(filepath, infostr, metric1_name, metric1_values, metric2_name, metric2_values, test_performance, test_bias):
    """
    Writes training configuration, per-epoch metrics, and final results to a log file.
    
    Args:
        filepath (str): Path to save the log file.
        infostr (str): The configuration string generated at the start of training.
        metric1_name (str): Name of the first metric (e.g., 'Train Loss').
        metric1_values (list): List of metric 1 values per epoch.
        metric2_name (str): Name of the second metric (e.g., 'Val NDCG').
        metric2_values (list): List of metric 2 values per epoch.
        test_performance (float): Final performance on the test set.
        test_bias (float): Final bias calculation on the test set.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"{filepath}_{timestamp}.txt", 'x') as f:
        # 1. Write the Header Info String
        f.write(infostr.strip() + '\n\n')
        
        # 2. Write the Per-Epoch Metrics Table
        f.write("================================================================================\n")
        f.write("                               PER EPOCH METRICS\n")
        f.write("================================================================================\n")
        # Format table headers
        f.write(f"{'Epoch':<10} | {metric1_name:<20} | {metric2_name:<20}\n")
        f.write("-" * 80 + "\n")
        
        # Determine number of epochs (handles case where lists might slightly differ in length if interrupted)
        epochs = max(len(metric1_values), len(metric2_values))
        
        # Write rows
        for i in range(epochs):
            m1 = f"{metric1_values[i]:.6f}" if i < len(metric1_values) else "N/A"
            m2 = f"{metric2_values[i]:.6f}" if i < len(metric2_values) else "N/A"
            f.write(f"{i+1:<10} | {m1:<20} | {m2:<20}\n")
            
        # 3. Write Final Test Results
        f.write("\n================================================================================\n")
        f.write("                               FINAL TEST RESULTS\n")
        f.write("================================================================================\n")
        f.write(f"Test Performance:       {test_performance:.6f}\n")
        f.write(f"Test Bias:              {test_bias:.6f}\n")
        f.write("================================================================================\n")

    print(f"Log saved successfully to {filepath}")

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std

def adv_net_testing(features, labels, device, latent_dim, label):
    num_men = sum(labels)
    features_len = len(labels)
    features = np.array(features, dtype=np.float32)
    lables = np.array(labels, dtype=np.float64)

    #create train test split for adversarial network
    from sklearn.model_selection import train_test_split
    features_train, features_test, lables_train, lables_test = train_test_split(features, lables, test_size=0.2, stratify=lables)

    #np.save('features_for_advnet_multvae_da', features)
    #np.save('labeles_for_advnet_multvae_da', lables)

    adv_model = ADV_net(200, 100, dropout=0.5)
    adv_optim = optim.Adam(
        adv_model.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    weights = torch.tensor([features_len/(2*(features_len-num_men)),features_len/(2*num_men)], dtype=torch.float32)
    weights = weights.to(device)
    adv_loss = nn.CrossEntropyLoss(weights)

    adv_model.to(device)
    for _ in range(100):
        for batch in range(np.ceil(features_train.shape[0]/128).astype(int)):
            
            x_data = torch.from_numpy(features_train[batch*128:np.min([batch*128+128, features_train.shape[0]])])
            y_data = torch.from_numpy(lables_train[batch*128:np.min([batch*128+128, features_train.shape[0]])]).long()

            x_data = x_data.to(device)
            y_data = y_data.to(device)

            adv_optim.zero_grad()
            predictions = adv_model(x_data)
            loss = adv_loss(predictions, y_data)

            loss.backward()
            adv_optim.step()
    b_acc = 0.0
    with torch.no_grad():
        features_test = torch.from_numpy(features_test).to(device)
        predictions_test = adv_model(features_test)
        from sklearn.metrics import balanced_accuracy_score

        # After training, get all test predictions
        all_preds = torch.argmax(predictions_test, dim=1).cpu().numpy()
        all_true = lables_test

        b_acc = balanced_accuracy_score(all_true, all_preds)
        #print(f"Standardized Balanced Accuracy for {label}: {b_acc}")

    return b_acc

def visualize_pca(features_for_pca, lables_for_visual, epochs, anneal_cap, b_acc, best_result):
    #create random sample of 1000 features for PCA visualization
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt

    index = np.array([random.randint(0, features_for_pca.shape[0]-1) for _ in range(1000)])
    random_sample = features_for_pca[index]
    random_label = lables_for_visual[index]

    random_sample = random_sample.reshape(-1, random_sample.shape[-1]) #features_for_pca.reshape(-1, features_for_pca.shape[-1])
    random_label = random_label.reshape(-1, random_label.shape[-1]) #lables_for_visual.reshape(-1, lables_for_visual.shape[-1])

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(random_sample)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=random_label, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA Visualization multVAE, epochs: {epochs}, anneal_cap: {anneal_cap}, balanced_acc: {b_acc:.4f}, best_ndcg@10: {best_result:.4f}')
    # make savefig not overwrite existing files
    
    
    filename = 'multvae_ADV_PCA.png'
    counter = 1
    while os.path.exists(filename):
        filename = f'multvae_ADV_PCA_{counter}.png'
        counter += 1
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def train(epochs, train_loader, train_user_info, test_loader=None, val_loader=None, val_user_info=None, test_user_info=None, anneal_steps=10000, anneal_cap=0.8, small_model=False, dropout=0.5, store_model=False, alpha=1.0, adv_net_dim=100, store_info=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    standard_model = [train_loader.dataset.num_items, 600, 200]
    small_model_dim = [train_loader.dataset.num_items, 500]
    model = MultVAE_ADV(standard_model, latent_dim=200, dropout=dropout, training=True, alpha=alpha, adv_net_dim=adv_net_dim)
    if small_model:
            model = MultVAE_ADV(small_model_dim, decoder_dims=[3416], latent_dim=200, dropout=dropout, training=True, alpha=1.0, adv_net_dim=100)
    model.to(device)
    total_anneal_steps = anneal_steps  # Anneal over ~20-50 epochs depending on dataset size
    anneal_cap = anneal_cap
    update_count = 0

    multVAEParams = nn.ParameterList()
    multVAEParams += model.encoder.parameters()
    multVAEParams += model.logvar_layer.parameters()
    multVAEParams += model.mu_layer.parameters()
    multVAEParams += model.decoder.parameters()

    optimizer = optim.Adam(
        multVAEParams,
        lr=1e-3,           # Learning rate
        weight_decay=0.0   # MultVAE usually relies on Dropout/KL-divergence for regularization, not L2
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    adv_optimizer = optim.Adam(
        model.adv_net.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )
    infostr = f"""
================================================================================
                            TRAINING CONFIGURATION
================================================================================
Model Type:             MultVAE_ADV
Encoder Dims:           {200}
Latent Dim:             200
Adversarial Net Dim:    {adv_net_dim}
Total Epochs:           {epochs}
Dropout Rate:           {dropout}
KL Anneal Steps:        {total_anneal_steps}
KL Anneal Cap:          {anneal_cap}
Adversarial Alpha:      {alpha}
Optimizer:              Adam (lr=1e-3, weight_decay=0.0)
================================================================================
"""
    men_count = train_user_info['gender'].sum()
    all_count = len(train_user_info)
    weights = torch.tensor([(men_count/all_count)/(1-(men_count/all_count)), 1.0], device=device, dtype=torch.float32)  # Adjust weights for each class if needed
    domain_loss = nn.CrossEntropyLoss(weight=weights)

    k = 0.5
    x0 = epochs / 4.0

    print('Starting training...')
    best_result = 0.0
    best_model = None
    least_bias = 1.0
    least_bias_model = None
    ndcg_metrics = []
    bacc_metrics = []
    for j in tqdm(range(epochs)):
        for i, (x_data, _, idx) in enumerate(train_loader):
            model.zero_grad()
            model.eval()
            x_data = x_data.to(device)
            recon_batch, mu, logvar, _ = model(x_data)

            real_domain = train_user_info.iloc[idx.numpy()]
            gender_array = real_domain['gender'].to_numpy()
            gender_tensor = torch.from_numpy(gender_array.copy()).long()
            gender_tensor = gender_tensor.to(device)

            for _ in range(3): # Train adversary a few extra times per VAE step
                adv_pred_inner = model.adv_net(model.grad_rev(mu.detach())) # Detach to not train encoder here
                gender_prediction_loss = domain_loss(adv_pred_inner, gender_tensor)
                adv_optimizer.zero_grad()
                gender_prediction_loss.backward()
                adv_optimizer.step()
            model.zero_grad()
            model.train()

            recon_batch, mu, logvar, adv_pred_outer = model(x_data)
            
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap
            update_count += x_data.size(0)
            log_probs = F.log_softmax(recon_batch, dim=1)

            current_alpha = alpha / (1 + np.exp(-k * (j - x0)))
            model.update_alpha(current_alpha)

            gender_prediction_loss = domain_loss(adv_pred_outer, gender_tensor)

            # Loss calculation
            MLL = torch.mean(-torch.sum(log_probs * x_data, dim=1))
            KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = MLL + anneal * KLD + gender_prediction_loss
            
            loss.backward()
            optimizer.step()
            adv_optimizer.step()
        scheduler.step()
            #print(f"MLL: {MLL.item()}, KLD: {KLD.item()}, ADV: {gender_prediction_loss.item()}")

        if val_loader is not None:
            model.eval()
            batch_evaluator = BatchEvaluator(metrics=["ndcg", "recall"], top_k=[10, 50])
            features_for_pca = []
            lables_for_visual = []
            with torch.no_grad():
                for (data, targets, val_idx) in val_loader:
                    len_batch = data.size(0)
                    data = data.to(device)
                    recon_batch, mu, logvar, _ = model(data)
                    recon_batch = recon_batch - (1000 * data)
                    batch_evaluator.eval_batch(np.arange(len_batch), recon_batch.cpu(), targets.cpu())
                    features_for_pca.extend(mu.cpu().detach().numpy().tolist())
                    lables_for_visual.extend(val_user_info.iloc[val_idx.numpy()]['gender'].tolist())
                result = batch_evaluator.get_results()
                ndcg_metrics.append(result.aggregated_metrics['ndcg@10'])
                #print(f'Validation Metrics after epoch {j+1}: {result.aggregated_metrics}')
                if result.aggregated_metrics['ndcg@10'] > best_result:  # Example threshold for early stopping
                    best_result = result.aggregated_metrics['ndcg@10']
                    best_model = model.state_dict()  # Save the best model weights
                    #print(f'New best model found at epoch {j+1} with NDCG@10: {best_result:.4f}')
            b_acc = adv_net_testing(features_for_pca, lables_for_visual, device=device, latent_dim=200, label=f"MultVAE_ADV, epoch:{j}")
            bacc_metrics.append(b_acc)
            if abs(b_acc-0.5) < abs(least_bias-0.5) and j+1 > epochs/3:
                #print(f"New lowest bias: {b_acc} at epoch {j+1}")
                least_bias = b_acc
                least_bias_model = model.state_dict()
            model.train()

    features_for_pca = []
    lables_for_visual = []
    #print(f"Best results: {best_result}, least bias: {least_bias}")
    #model.load_state_dict(least_bias_model)  # Load the best model weights before extracting features
    model.eval()
    for _, (x_data, _, idx) in enumerate(train_loader):
            x_data = x_data.to(device)
            _, mu, logvar, _ = model(x_data)
            #sample_z = reparameterize(mu=mu, logvar=logvar)
            #sample_z = sample_z.to(device)
            features_for_pca.extend(mu.cpu().detach().numpy().tolist())
            lables_for_visual.extend(train_user_info.iloc[idx.numpy()]['gender'].tolist())


    # visualize PCA with gender variable
    num_men = sum(lables_for_visual)
    features_len = len(lables_for_visual)
    #lables_for_visual = [[1, 0] if x == 0 else [0, 1] for x in lables_for_visual]

    features_for_pca = np.array(features_for_pca, dtype=np.float32)
    lables_for_visual = np.array(lables_for_visual, dtype=np.float64)

    #create train test split for adversarial network
    from sklearn.model_selection import train_test_split
    features_train, features_test, lables_train, lables_test = train_test_split(features_for_pca, lables_for_visual, test_size=0.2, stratify=lables_for_visual)

    # TRAIN ADV-net
    adv_model = ADV_net(200, 100, dropout=0.5)
    adv_optim = optim.Adam(
        adv_model.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    weights = torch.tensor([features_len/(2*(features_len-num_men)),features_len/(2*num_men)], dtype=torch.float32)
    weights = weights.to(device)
    adv_loss = torch.nn.CrossEntropyLoss(weights)

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = features_train#scaler.fit_transform(features_train)
    X_test_scaled = features_test#scaler.transform(features_test)

    adv_model.to(device)
    for _ in tqdm(range(100)):
        permuted_indices = np.random.permutation(X_train_scaled.shape[0])
        training_samples = X_train_scaled[permuted_indices]
        training_labels = lables_train[permuted_indices]
        for batch in range(np.ceil(X_train_scaled.shape[0]/128).astype(int)):
            
            x_data = torch.from_numpy(training_samples[batch*128:np.min([batch*128+128, training_samples.shape[0]])])
            y_data = torch.from_numpy(training_labels[batch*128:np.min([batch*128+128, training_labels.shape[0]])]).long()

            x_data = x_data.to(device)
            y_data = y_data.to(device)

            #print(x_data.type())
            #print(y_data.type())

            adv_optim.zero_grad()
            predictions = adv_model(x_data)
            #print(F.softmax(predictions)[0])
            #print(predictions.type())
            loss = adv_loss(predictions, y_data)

            loss.backward()
            adv_optim.step()
    b_acc = 0.0
    with torch.no_grad():
        X_test_scaled = torch.from_numpy(X_test_scaled).to(device)
        predictions_test = adv_model(X_test_scaled)

        all_preds = torch.argmax(predictions_test, dim=1).cpu().numpy()
        all_true = lables_test

        b_acc = balanced_accuracy_score(all_true, all_preds)
        #print(f"Standardized Balanced Accuracy: {b_acc}")

    

    # check performance on test set after training is complete
    #model.load_state_dict(least_bias_model)  # Load the best model weights before testing
    dataset = 'ml1m'
    if train_loader.dataset.num_items > 4000:
        dataset = 'ekstrabladet'
    if train_loader.dataset.num_items > 10000:
        dataset = 'lfmdemobias'
    if store_model:
        PATH = f'./ml1m_multvae_ADV_{dataset}.pth'
        torch.save(model.state_dict(), PATH)
    if test_loader is not None:
        model.eval()
        batch_evaluator = BatchEvaluator(metrics=["ndcg", "recall"], top_k=[10, 50])
        with torch.no_grad():
            for (data, targets, _) in test_loader:
                len_batch = data.size(0)
                data = data.to(device)
                recon_batch, mu, logvar, _ = model(data)
                recon_batch = recon_batch - (1000 * data)
                batch_evaluator.eval_batch(np.arange(len_batch), recon_batch.cpu(), targets.cpu())
                
            #print(f'Test Metrics: {batch_evaluator.get_results().aggregated_metrics}')
            visualize_pca(features_for_pca=features_for_pca, lables_for_visual=lables_for_visual, epochs=epochs, anneal_cap=anneal_cap, b_acc=b_acc, best_result=batch_evaluator.get_results(reset_state=False).aggregated_metrics["ndcg@10"])
            save_training_log(f"multVAE_ADV", infostr, "balanced accuracy", bacc_metrics, "NDCG@10", ndcg_metrics, batch_evaluator.get_results(reset_state=False).aggregated_metrics["ndcg@10"], b_acc)
            return b_acc, batch_evaluator.get_results().aggregated_metrics if test_loader is not None else None
        
    return b_acc, None
