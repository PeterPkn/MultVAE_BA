import torch
from torch import nn
from torch.nn import functional as F

class MultVAE(nn.Module):
    def __init__(self, dims, latent_dim=200, dropout=0.5, training=True):
        super(MultVAE, self).__init__()
        self.enc_dims = dims
        self.dec_dims = dims[::-1]
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout)
        self.training = training

        self.encoder = nn.ModuleList()
        for i in range(len(self.enc_dims) - 1):
            self.encoder.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder.append(nn.Tanh())

        self.mu_layer = nn.Linear(self.enc_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(self.enc_dims[-1], latent_dim)

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(latent_dim, self.dec_dims[0]))
        self.decoder.append(nn.Tanh())
        for i in range(len(self.dec_dims) - 1):
            self.decoder.append(nn.Linear(self.dec_dims[i], self.dec_dims[i + 1]))
        
    def encoder_forward(self, x):
        if self.training:
            x = self.dropout(x)
        for layer in self.encoder:
            x = layer(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decoder_forward(self, z):
        for layer in self.decoder:
            z = layer(z)
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder_forward(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder_forward(z)
        return recon_x, mu, logvar