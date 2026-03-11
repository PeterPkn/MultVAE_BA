import os
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
from tqdm import tqdm

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

# input -> 500 -> 200 (latent) -> output
# testen von BAcc auf MultVAE (ohne CLUB) DONE
# PCA auf MultVAE oder t-SNE, UMAP DONE
# erste 10-20 epochen ohne CLUB ANNEALING DONE
# domain encoder balanced accuracy track DONE

def train(epochs, train_loader, train_user_info, test_loader=None, val_loader=None, anneal_steps=10000, anneal_cap=0.8, club_weight=5000.0, prioritize_bias=False, latent_dim_domain=200):
    print("Training multVAE with domain adaptation...")
    print(f"Club weight: {club_weight}, Anneal steps: {anneal_steps}, Anneal cap: {anneal_cap}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    latent_dim = 200

    model = MultVAE_DA([3416, 600, 200], latent_dim=latent_dim, dropout=0.5, training=True, latent_dim_domain=latent_dim_domain)
    mi_model = MI_net(200, 300, latent_dim_domain)
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
        lr=2e-3,
        weight_decay=0.0
    )

    #train_user_info['gender'].map({'M': 1, 'F': 0})
    men_count = train_user_info['gender'].value_counts().get('M', 0)
    all_count = len(train_user_info)
    weights = torch.tensor([(men_count/all_count)/(1-(men_count/all_count)), 1.0], device=device, dtype=torch.float32)  # Adjust weights for each class if needed

    domain_loss = nn.CrossEntropyLoss(weight=weights)

    print('Starting training...')
    best_result = 0.0
    best_epoch = 0
    best_model = None
    last_model = None
    features_for_pca = []
    lables_for_visual = []
    for j in tqdm(range(epochs)):
        avg_gender_loss = 0.0
        bound_sum = 0.0
        kld_sum = 0.0
        mll_sum = 0.0

        for _, (x_data, _, idx) in enumerate(train_loader):
            x_data = x_data.to(device)

            #rand = random.randint(0, x_data.shape[0]-1)
            # maximize log-likelyhood for MI-estimation, 
            # CURRENT METHOD: MI_net predicts domain distribution from feature sample -> 
            # calculate probability of domain sample given predicted distribution
            for _ in range(5):
                mi_optim.zero_grad()
                _, mu_mipass, logvar_mipass, _, mu_domain_mipass, logvar_domain_mipass = model(x_data)
                sample_z_mipass = reparameterize(mu=mu_mipass, logvar=logvar_mipass).detach()  # Detach to prevent gradients from flowing into the VAE
                sample_z_mipass = sample_z_mipass.to(device)
                #print(sample_z_mipass.shape)
                #features_for_pca.extend(sample_z_mipass.cpu().detach().numpy().tolist())
                #sample_domain_z_mipass = reparameterize(mu=mu_domain_mipass, logvar=logvar_domain_mipass).detach()
                target_domain = mu_domain_mipass.detach()
                domain_pred_mu_mipass, domain_pred_logvar_mipass = mi_model(sample_z_mipass)
                mi_loss = -torch.mean(log_likelihood(target_domain, domain_pred_mu_mipass, domain_pred_logvar_mipass)) # negative log likelihood
                mi_loss.backward()
                mi_optim.step()

            context_optim.zero_grad()
            recon_batch_featureoptim, mu_featureoptim, logvar_featureoptim, _, mu_domain_featureoptim, logvar_domain_featureoptim = model(x_data)
            # Compute VAE loss
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, anneal_cap * update_count / total_anneal_steps)
                club_anneal = 0 if j < 40 else min(1., 1. * (j - 40) / (epochs - 40))  # Linearly increase CLUB weight after 40 epochs
            else:
                anneal = anneal_cap
                club_anneal = club_weight
            update_count += x_data.size(0)
            #print(f"Processed samples: {update_count}")  # count number of samples processed


            log_probs = F.log_softmax(recon_batch_featureoptim, dim=1)

            # Loss calculation
            MLL = torch.mean(-torch.sum(log_probs * x_data, dim=1))
            KLD = torch.mean(-0.5 * torch.sum(1 + logvar_featureoptim - mu_featureoptim.pow(2) - logvar_featureoptim.exp(), dim=1))

            # Calculate KLD for both spaces to keep them anchored around 0
            #KLD_c = torch.mean(-0.5 * torch.sum(1 + logvar_featureoptim - mu_featureoptim.pow(2) - logvar_featureoptim.exp(), dim=1))
            #KLD_d = torch.mean(-0.5 * torch.sum(1 + logvar_domain_featureoptim - mu_domain_featureoptim.pow(2) - logvar_domain_featureoptim.exp(), dim=1))
            
            # Combine them
            #KLD = KLD_c + KLD_d

            # log q(yi|xi)

            # for each sample:
            # calculate log q(yi|xi)
            #sample_z_featureoptim = reparameterize(mu=mu_featureoptim, logvar=logvar_featureoptim)
            #sample_domain_z_featureoptim = reparameterize(mu=mu_domain_featureoptim, logvar=logvar_domain_featureoptim)
            target_domain_featureoptim = mu_domain_featureoptim.detach()
            domain_pred_mu_featureoptim, domain_pred_logvar_featureoptim = mi_model(mu_featureoptim)
            batch_size = mu_featureoptim.shape[0]

            z_expanded = target_domain_featureoptim.unsqueeze(1)
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
            bound = torch.clamp(bound, min=0.0)

            bound_sum += bound.item()
            kld_sum += KLD.item()
            mll_sum += MLL.item()
            #sample_domain_z = reparameterize(mu=mu_domain, logvar=logvar_domain)
            #domain_pred_mu, domain_pred_logvar = mi_model(sample_z)
            #q_yi_xi = log_likelihood(sample_domain_z, domain_pred_mu, domain_pred_logvar)
            ## for every sample:
            ## calculate sum over j log q(yj|xi) with j != i
            #q_yj_xi = torch.zeros([batch_size, 1])
            #for idx_x, _ in enumerate(sample_z):
            #    q_yj_xi[idx_x] = torch.mean(log_likelihood(sample_domain_z, domain_pred_mu[idx_x] * batch_size, domain_pred_logvar[idx_x] * batch_size))

           # #bound = torch.mean(q_yi_xi - q_yj_xi)


            loss_feature = MLL + anneal * KLD + (club_anneal * club_weight) * bound
            #print(bound)
            loss_feature.backward()
            context_optim.step()

            model.zero_grad()
            mi_model.zero_grad()
            domain_optim.zero_grad()
            _, _, _, domain_predictions_domainoptim, mu_domain_domainoptim, logvar_domain_domainoptim = model(x_data) # 0 is female, 1 is male
            real_domain = train_user_info.iloc[idx.numpy()]
            gender_map = {'M': 1, 'F': 0}
            gender_array = real_domain['gender'].map(gender_map).to_numpy()
            #lables_for_visual.extend(gender_array)
            gender_tensor = torch.from_numpy(gender_array.copy())
            gender_tensor = gender_tensor.to(device)

            #print(domain_predictions_domainoptim.type())
            #print(gender_tensor.type())

            KLD_d = torch.mean(-0.5 * torch.sum(1 + logvar_domain_domainoptim - mu_domain_domainoptim.pow(2) - logvar_domain_domainoptim.exp(), dim=1))
            #l2_reg = torch.mean(mu_domain_domainoptim.pow(2))

            loss = domain_loss(domain_predictions_domainoptim, gender_tensor) #+ anneal * KLD_d #0.05 * l2_reg

            #print(loss.item())

            avg_gender_loss += loss.item()
            
            loss.backward()
            domain_optim.step()
        print(f"\nEpoch {j+1}, Batch MI Bound: {bound_sum/len(train_loader):.4f}, KLD: {kld_sum/len(train_loader):.4f}, MLL: {mll_sum/len(train_loader):.4f}, anneal: {anneal:.4f}, club_anneal: {(club_anneal * club_weight):.4f}")
        print(f'Average gender prediction loss after epoch {j+1}: {avg_gender_loss/len(train_loader):.4f}')
            

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
                    best_epoch = j+1
                    best_model = model.state_dict()  # Save the best model weights
                    #print(f'New best model found at epoch {j+1} with NDCG@10: {best_result:.4f}')
            model.train()

    last_model = model.state_dict()  # Save the last model weights after training is complete

    # get some random samples from the latent space to train the adversarial network on
    print(f"Best epoch: {best_epoch}")
    model.load_state_dict(best_model)  # Load the best model weights before extracting features
    if prioritize_bias:
        print("Prioritizing bias in feature extraction by loading the last model weights...")
        model.load_state_dict(last_model)  # Load the last model weights if prioritizing bias
    model.eval()
    for _, (x_data, _, idx) in enumerate(train_loader):
            x_data = x_data.to(device)
            _, mu, logvar, _, _, _ = model(x_data)
            #sample_z = reparameterize(mu=mu, logvar=logvar)
            #sample_z = sample_z.to(device)
            features_for_pca.extend(mu.cpu().detach().numpy().tolist())
            lables_for_visual.extend(train_user_info.iloc[idx.numpy()]['gender'].map({'M': 1, 'F': 0}).tolist())


    num_men = sum(lables_for_visual)
    features_len = len(lables_for_visual)
    #lables_for_visual = [[1, 0] if x == 0 else [0, 1] for x in lables_for_visual]

    features_for_pca = np.array(features_for_pca, dtype=np.float32)
    lables_for_visual = np.array(lables_for_visual, dtype=np.float64)

    #create train test split for adversarial network
    from sklearn.model_selection import train_test_split
    features_train, features_test, lables_train, lables_test = train_test_split(features_for_pca, lables_for_visual, test_size=0.2, stratify=lables_for_visual)

    np.save('features_for_advnet_multvae_da', features_for_pca)
    np.save('labeles_for_advnet_multvae_da', lables_for_visual)

    # TRAIN ADV-net

    adv_model = ADV_net(200, 100)
    adv_optim = optim.Adam(
        adv_model.parameters(),
        lr=1e-3,
        weight_decay=0.0
    )

    weights = torch.tensor([features_len/(2*(features_len-num_men)),features_len/(2*num_men)], dtype=torch.float32)
    weights = weights.to(device)
    adv_loss = nn.CrossEntropyLoss(weights)


    #print(features_train.shape)
    #print(lables_train.shape)
    adv_model.to(device)
    for _ in tqdm(range(100)):
        for batch in range(np.ceil(features_train.shape[0]/128).astype(int)):
            
            x_data = torch.from_numpy(features_train[batch*128:np.min([batch*128+128, features_train.shape[0]])])
            y_data = torch.from_numpy(lables_train[batch*128:np.min([batch*128+128, features_train.shape[0]])]).long()

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
        features_test = torch.from_numpy(features_test).to(device)
        predictions_test = adv_model(features_test)
        from sklearn.metrics import balanced_accuracy_score

        # After training, get all test predictions
        all_preds = torch.argmax(predictions_test, dim=1).cpu().numpy()
        all_true = lables_test #torch.argmax(torch.from_numpy(lables_test), dim=1).numpy()

        b_acc = balanced_accuracy_score(all_true, all_preds)
        print(f"Standardized Balanced Accuracy: {b_acc}")
        # 0.5 is random chance, 1.0 is perfect bias, 0.0 is perfectly wrong
    # TODO: train user etc. blabblablabla


    print(features_for_pca.shape)
    print(lables_for_visual.shape)

    #create random sample of 200 features for PCA visualization
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
    plt.title(f'PCA Visualization, epochs: {epochs}, club_weight: {club_weight}, anneal_cap: {anneal_cap}, balanced_acc: {b_acc:.4f}, best_epoch: {best_epoch}, best_ndcg@10: {best_result:.4f}, latent dim domain size: {latent_dim_domain}')
    # make savefig not overwrite existing files
    
    
    filename = 'multvae_da_PCA.png'
    counter = 1
    while os.path.exists(filename):
        filename = f'multvae_da_PCA_{counter}.png'
        counter += 1
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()

    # check performance on test set after training is complete
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