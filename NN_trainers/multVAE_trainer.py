import os
import random
import torch
from NeuralNetworks.multVAE import MultVAE
import torch.optim as optim
import torch.nn.functional as F
from rmet import BatchEvaluator
import numpy as np
from tqdm import tqdm
from NeuralNetworks.ADV_net import ADV_net

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std


def train(epochs, train_loader, test_loader=None, val_loader=None, train_user_info=None, anneal_steps=10000, anneal_cap=0.8):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MultVAE([3416, 600, 200], latent_dim=200, dropout=0.5, training=True)
    model.to(device)
    total_anneal_steps = anneal_steps  # Anneal over ~20-50 epochs depending on dataset size
    anneal_cap = anneal_cap
    update_count = 0
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,           # Learning rate
        weight_decay=0.0   # MultVAE usually relies on Dropout/KL-divergence for regularization, not L2
    )
    print('Starting training...')
    best_result = 0.0
    best_model = None
    for j in tqdm(range(epochs)):
        for i, (x_data, _, _) in enumerate(train_loader):
            x_data = x_data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(x_data)
            # Compute VAE loss
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap
            update_count += x_data.size(0)  # count number of samples processed
            log_probs = F.log_softmax(recon_batch, dim=1)

            # Loss calculation
            MLL = torch.mean(-torch.sum(log_probs * x_data, dim=1))
            KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = MLL + anneal * KLD
            
            loss.backward()
            optimizer.step()
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
                    recon_batch, mu, logvar = model(data)
                    recon_batch = recon_batch - (1000 * data)
                    batch_evaluator.eval_batch(np.arange(len_batch), recon_batch.cpu(), targets.cpu())
                result = batch_evaluator.get_results()
                #print(f'Validation Metrics after epoch {j+1}: {result.aggregated_metrics}')
                if result.aggregated_metrics['ndcg@10'] > best_result:  # Example threshold for early stopping
                    best_result = result.aggregated_metrics['ndcg@10']
                    best_model = model.state_dict()  # Save the best model weights
                    #print(f'New best model found at epoch {j+1} with NDCG@10: {best_result:.4f}')
            model.train()

    features_for_pca = []
    lables_for_visual = []
    for _, (x_data, _, idx) in enumerate(train_loader):
            x_data = x_data.to(device)
            _, mu, logvar = model(x_data)
            #sample_z = reparameterize(mu=mu, logvar=logvar)
            #sample_z = sample_z.to(device)
            features_for_pca.extend(mu.cpu().detach().numpy().tolist())
            lables_for_visual.extend(train_user_info.iloc[idx.numpy()]['gender'].map({'M': 1, 'F': 0}).tolist())


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

    indices = np.random.permutation(len(features_train))
    features_train = features_train[indices]
    lables_train = lables_train[indices]

    features_test = features_for_pca[features_len-1000:]
    lables_test = lables_for_visual[features_len-1000:]

    np.save('features_for_advnet_multvae', features_for_pca)
    np.save('labeles_for_advnet_multvae', lables_for_visual)

    # TRAIN ADV-net

    adv_model = ADV_net(200, 100)
    adv_optim = optim.Adam(
        adv_model.parameters(),
        lr=1e-3,
        weight_decay=0.0
    )

    weights = torch.tensor([features_len/(2*(features_len-num_men)),features_len/(2*num_men)], dtype=torch.float32)
    weights = weights.to(device)
    adv_loss = torch.nn.CrossEntropyLoss(weights)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_train)
    X_test_scaled = scaler.transform(features_test)

    # Use a simple Sklearn Logistic Regression
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train_scaled, lables_train)
    sk_preds = clf.predict(X_test_scaled)
    print(f"Sklearn BAcc: {balanced_accuracy_score(lables_test, sk_preds)}")

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
    plt.title('PCA Visualization')
    # make savefig not overwrite existing files
    
    
    filename = 'multvae_da_PCA.png'
    counter = 1
    while os.path.exists(filename):
        filename = f'multvae_da_PCA_{counter}.png'
        counter += 1
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # check performance on test set after training is complete
    model.load_state_dict(best_model)  # Load the best model weights before testing
    if test_loader is not None:
        model.eval()
        batch_evaluator = BatchEvaluator(metrics=["ndcg", "recall"], top_k=[10, 50])
        with torch.no_grad():
            for (data, targets, _) in test_loader:
                len_batch = data.size(0)
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                recon_batch = recon_batch - (1000 * data)
                batch_evaluator.eval_batch(np.arange(len_batch), recon_batch.cpu(), targets.cpu())
                
            print(f'Test Metrics: {batch_evaluator.get_results().aggregated_metrics}')

    PATH = './ml1m_multvae.pth'
    torch.save(model.state_dict(), PATH)