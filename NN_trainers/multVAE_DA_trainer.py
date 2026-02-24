import torch
from NeuralNetworks.multVAE_DA import MultVAE_DA
from NeuralNetworks.MI_estimation_NN import MI_net
from NeuralNetworks.ADV_net import ADV_net
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from rmet import BatchEvaluator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset
import random

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std

def log_likelihood(y, mu, logvar):
    # y: [batch_size, dimensions]
    # mu: [batch_size, dimensions]
    # logvar: [batch_size, dimensions]
    
    # This calculation happens element-wise, then sums across dimensions
    # to return a [batch_size] vector of scalars.
    return -0.5 * torch.sum(
        logvar + (y - mu)**2 / torch.exp(logvar) + np.log(2 * np.pi), 
        dim=1
    )

def train(epochs, train_loader, train_user_info, test_loader=None, val_loader=None, anneal_steps=10000, anneal_cap=0.8, club_weight=5000.0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    latent_dim = 200

    model = MultVAE_DA([3416, 600, 200], latent_dim=latent_dim, dropout=0.5, training=True)
    mi_model = MI_net(200, 70)
    mi_model.to(device)
    model.to(device)
    total_anneal_steps = anneal_steps  # Anneal over ~20-50 epochs depending on dataset size
    anneal_cap = anneal_cap
    update_count = 0

    context_variables = nn.ParameterList()
    context_variables += model.encoder.parameters()
    context_variables += model.logvar_layer.parameters()
    context_variables += model.mu_layer.parameters()
    context_variables += model.decoder.parameters()

    domain_variables = nn.ParameterList()
    domain_variables += model.encoder_domain.parameters()
    domain_variables += model.logvar_layer_domain.parameters()
    domain_variables += model.mu_layer_domain.parameters()
    domain_variables += model.decoder_domain.parameters()

    mi_variables = mi_model.parameters()

    context_optim = optim.Adam(
        context_variables,
        lr=1e-3,
        weight_decay=0.0
    )
    domain_optim = optim.Adam(
        domain_variables,
        lr=1e-3,
        weight_decay=0.0
    )
    mi_optim = optim.Adam(
        mi_variables,
        lr=1e-3,
        weight_decay=0.0
    )

    domain_loss = nn.CrossEntropyLoss()

    print('Starting training...')
    best_result = 0.0
    best_model = None
    features_for_pca = []
    lables_for_visual = []
    for j in range(epochs):
        for _, (x_data, _, idx) in enumerate(train_loader):
            x_data = x_data.to(device)

            #rand = random.randint(0, x_data.shape[0]-1)
            # maximize log-likelyhood for MI-estimation, 
            # CURRENT METHOD: MI_net predicts domain distribution from feature sample -> 
            # calculate probability of domain sample given predicted distribution
            mi_optim.zero_grad()
            _, mu_mipass, logvar_mipass, _, mu_domain_mipass, logvar_domain_mipass = model(x_data)
            sample_z_mipass = reparameterize(mu=mu_mipass, logvar=logvar_mipass)
            sample_z_mipass = sample_z_mipass.to(device)
            #print(sample_z_mipass.shape)
            features_for_pca.extend(sample_z_mipass.cpu().detach().numpy().tolist())
            sample_domain_z_mipass = reparameterize(mu=mu_domain_mipass, logvar=logvar_domain_mipass)
            domain_pred_mu_mipass, domain_pred_logvar_mipass = mi_model(sample_z_mipass)
            mi_loss = -torch.mean(log_likelihood(sample_domain_z_mipass, domain_pred_mu_mipass, domain_pred_logvar_mipass)) # negative log likelihood
            mi_loss.backward()
            mi_optim.step()

            recon_batch_featureoptim, mu_featureoptim, logvar_featureoptim, domain_prediction_featureoptim, mu_domain_featureoptim, logvar_domain_featureoptim = model(x_data)
            # Compute VAE loss
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap
            update_count += x_data.size(0)  # count number of samples processed


            log_probs = F.log_softmax(recon_batch_featureoptim, dim=1)

            # Loss calculation
            MLL = torch.mean(-torch.sum(log_probs * x_data, dim=1))
            KLD = torch.mean(-0.5 * torch.sum(1 + logvar_featureoptim - mu_featureoptim.pow(2) - logvar_featureoptim.exp(), dim=1))

            # log q(yi|xi)

            # for each sample:
            # calculate log q(yi|xi)
            sample_z_featureoptim = reparameterize(mu=mu_featureoptim, logvar=logvar_featureoptim)
            sample_domain_z_featureoptim = reparameterize(mu=mu_domain_featureoptim, logvar=logvar_domain_featureoptim)
            domain_pred_mu_featureoptim, domain_pred_logvar_featureoptim = mi_model(sample_z_featureoptim)           
            batch_size = sample_z_featureoptim.shape[0]

            z_expanded = sample_domain_z_featureoptim.unsqueeze(1) 
            mu_expanded = domain_pred_mu_featureoptim.unsqueeze(0)
            logvar_expanded = domain_pred_logvar_featureoptim.unsqueeze(0)

            # Calculate pairwise log-likelihoods: [B, B] matrix
            pairwise_ll = -0.5 * torch.sum(
                logvar_expanded + (z_expanded - mu_expanded)**2 / torch.exp(logvar_expanded) + np.log(2 * np.pi), 
                dim=2
            )

            # q_yi_xi is the diagonal (where i == j)
            q_yi_xi = torch.diag(pairwise_ll)

            # q_yj_xi is the mean of the off-diagonals
            row_sums = torch.sum(pairwise_ll, dim=1)
            q_yj_xi = (row_sums - q_yi_xi) / (batch_size - 1)

            bound = torch.mean(q_yi_xi - q_yj_xi)

            #sample_domain_z = reparameterize(mu=mu_domain, logvar=logvar_domain)
            #domain_pred_mu, domain_pred_logvar = mi_model(sample_z)
            #q_yi_xi = log_likelihood(sample_domain_z, domain_pred_mu, domain_pred_logvar)
            ## for every sample:
            ## calculate sum over j log q(yj|xi) with j != i
            #q_yj_xi = torch.zeros([batch_size, 1])
            #for idx_x, _ in enumerate(sample_z):
            #    q_yj_xi[idx_x] = torch.mean(log_likelihood(sample_domain_z, domain_pred_mu[idx_x] * batch_size, domain_pred_logvar[idx_x] * batch_size))

           # #bound = torch.mean(q_yi_xi - q_yj_xi)

            loss = MLL + anneal * KLD + club_weight * bound
            
            loss.backward()
            context_optim.step()

            model.zero_grad()
            domain_optim.zero_grad()
            _, _, _, domain_predictions_domainoptim, _, _ = model(x_data) # 0 is female, 1 is male
            real_domain = train_user_info.iloc[idx.numpy()]
            gender_map = {'M': 1, 'F': 0}
            gender_array = real_domain['gender'].map(gender_map).to_numpy()
            lables_for_visual.extend(gender_array)
            gender_tensor = torch.from_numpy(gender_array.copy()).float()
            gender_tensor = gender_tensor.to(device)
            loss = domain_loss(domain_predictions_domainoptim, torch.unsqueeze(gender_tensor, 1))
            loss.backward()
            domain_optim.step()
            

            #if i % 20 == 19:               
                #print(f'Epoch: {j+1}, Batch: {i+1}, anneal: {anneal:.4f}')
        # check performance on validation set after each epoch
        if val_loader is not None:
            model.eval()
            batch_evaluator = BatchEvaluator(metrics=["ndcg", "recall"], top_k=[10, 50])
            with torch.no_grad():
                for (data, targets, _) in val_loader:
                    len_batch = data.size(0)
                    data = data.to(device)
                    recon_batch_val, _, _, _, _, _ = model(data)
                    recon_batch_val = recon_batch_val - (1000 * data)
                    batch_evaluator.eval_batch(np.arange(len_batch), recon_batch_val.cpu(), targets.cpu())
                result = batch_evaluator.get_results()
                #print(f'Validation Metrics after epoch {j+1}: {result.aggregated_metrics}')
                if result.aggregated_metrics['ndcg@10'] > best_result:  # Example threshold for early stopping
                    best_result = result.aggregated_metrics['ndcg@10']
                    best_model = model.state_dict()  # Save the best model weights
                    print(f'New best model found at epoch {j+1} with NDCG@10: {best_result:.4f}')
            model.train()

    # visualize PCA with gender variable
    num_men = sum(lables_for_visual)
    majority_class_percentage = num_men / len(lables_for_visual)
    #lables_for_visual = [[1, 0] if x == 0 else [0, 1] for x in lables_for_visual]

    features_for_pca = np.array(features_for_pca, dtype=np.float32)
    lables_for_visual = np.array(lables_for_visual, dtype=np.float64)

    #print(lables_for_visual[0:10])

    features_len = features_for_pca.shape[0]

    features_train = features_for_pca[0:features_len-1000]
    lables_train = lables_for_visual[0:features_len-1000]

    features_test = features_for_pca[features_len-1000:]
    lables_test = lables_for_visual[features_len-1000:]

    np.save('features_for_advnet', features_for_pca)
    np.save('labeles_for_advnet', lables_for_visual)

    # TRAIN ADV-net

    adv_model = ADV_net(200, 100)
    adv_optim = optim.Adam(
        adv_model.parameters(),
        lr=1e-3,
        weight_decay=0.0
    )
    adv_loss = nn.CrossEntropyLoss(torch.tensor([majority_class_percentage/(1-majority_class_percentage), majority_class_percentage/majority_class_percentage], dtype=torch.float32))


    #print(features_train.shape)
    #print(lables_train.shape)
    for _ in range(10):
        for batch in range(np.ceil(features_train.shape[0]/128).astype(int)):
            
            x_data = torch.from_numpy(features_train[batch*128:np.min([batch*128+128, features_train.shape[0]])])
            y_data = torch.from_numpy(lables_train[batch*128:np.min([batch*128+128, features_train.shape[0]])]).long()

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
        predictions_test = adv_model(torch.from_numpy(features_test))
        from sklearn.metrics import balanced_accuracy_score

        # After training, get all test predictions
        all_preds = torch.argmax(predictions_test, dim=1).cpu().numpy()
        all_true = lables_test #torch.argmax(torch.from_numpy(lables_test), dim=1).numpy()

        b_acc = balanced_accuracy_score(all_true, all_preds)
        print(f"Standardized Balanced Accuracy: {b_acc}")
        # 0.5 is random chance, 1.0 is perfect bias, 0.0 is perfectly wrong
    # TODO: train user etc. blabblablabla


    # print(features_for_pca.shape)
    # print(lables_for_visual.shape)

    # features_for_pca = features_for_pca.reshape(-1, features_for_pca.shape[-1])
    # lables_for_visual = lables_for_visual.reshape(-1, lables_for_visual.shape[-1])

    # pca = PCA(n_components=2)
    # reduced_features = pca.fit_transform(features_for_pca)

    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=lables_for_visual, cmap='viridis', alpha=0.7)
    # plt.colorbar(scatter, label='Class Labels')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('PCA Visualization of ResNet50 Features')
    # plt.show()

    # check performance on test set after training is complete
    model.load_state_dict(best_model)  # Load the best model weights before testing
    if test_loader is not None:
        model.eval()
        batch_evaluator = BatchEvaluator(metrics=["ndcg", "recall"], top_k=[10, 50])
        with torch.no_grad():
            for (data, targets, _) in test_loader:
                len_batch = data.size(0)
                data = data.to(device)
                recon_batch, mu, logvar, _, _, _ = model(data)
                recon_batch = recon_batch - (1000 * data)
                batch_evaluator.eval_batch(np.arange(len_batch), recon_batch.cpu(), targets.cpu())
                
            all_results = batch_evaluator.get_results().aggregated_metrics
            print(f'Test Metrics: {all_results}')

    PATH = './ml1m_multvae_DA.pth'
    torch.save(model.state_dict(), PATH)

    return b_acc, all_results["ndcg@10"]