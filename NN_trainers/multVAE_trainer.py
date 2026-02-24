import torch
from NeuralNetworks.multVAE import MultVAE
import torch.optim as optim
import torch.nn.functional as F
from rmet import BatchEvaluator
import numpy as np

def train(epochs, train_loader, test_loader=None, val_loader=None, anneal_steps=10000, anneal_cap=0.8):
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
    for j in range(epochs):
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
                    print(f'New best model found at epoch {j+1} with NDCG@10: {best_result:.4f}')
            model.train()

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