import torch
from torch import nn
from torch.nn import functional as F

class MultVAE_DA(nn.Module):
    def __init__(self, dims, dec_dims = None, latent_dim=200, latent_dim_domain=2, dropout=0.5, training=True):
        super(MultVAE_DA, self).__init__()
        self.enc_dims = dims
        self.dec_dims = dec_dims if dec_dims is not None else dims[::-1]
        self.latent_dim = latent_dim
        self.latent_dim_domain = latent_dim_domain
        self.dropout = nn.Dropout(dropout)
        self.training = training

        # TODO: The decoder currently has the same architecture as the encoder, but it could be different. Maybe we can use a simpler decoder for the domain adaptation task, since we only want to reconstruct the domain labels?

        # Encoder and Decoder for the main task
        self.encoder = nn.ModuleList()
        for i in range(len(self.enc_dims) - 1):
            self.encoder.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder.append(nn.Tanh())

        self.mu_layer = nn.Linear(self.enc_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(self.enc_dims[-1], latent_dim)

        self.decoder = nn.ModuleList()
        self.decoder.append(nn.Linear(self.latent_dim, self.dec_dims[0]))
        for i in range(len(self.dec_dims) - 1):
            self.decoder.append(nn.Tanh())
            self.decoder.append(nn.Linear(self.dec_dims[i], self.dec_dims[i + 1]))

        # Encoder and Decoder for the domain adaptation task
        self.encoder_domain = nn.ModuleList()
        for i in range(len(self.enc_dims) - 1):
            self.encoder_domain.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_domain.append(nn.Tanh())

        self.mu_layer_domain = nn.Linear(self.enc_dims[-1], self.latent_dim_domain)
        self.logvar_layer_domain = nn.Linear(self.enc_dims[-1], self.latent_dim_domain)

        self.decoder_domain = nn.ModuleList()
        self.decoder_domain.append(nn.Linear(self.latent_dim_domain, self.dec_dims[0]))
        self.decoder_domain.append(nn.Tanh())
        self.decoder_domain.append(nn.Linear(self.dec_dims[0], 100))
        self.decoder_domain.append(nn.Tanh())
        self.decoder_domain.append(nn.Linear(100, 2))
        #self.decoder_domain.append(nn.Sigmoid())
        #for i in range(len(self.dec_dims) - 1):
        #    self.decoder_domain.append(nn.Linear(self.dec_dims[i], self.dec_dims[i + 1]))
    
    def eval(self):
        self.training = False
        return super().eval()
    
    def train(self, mode=True):
        self.training = mode
        return super().train(mode)

    def encoder_forward(self, x):
        if self.training:
            x = self.dropout(x)
        for layer in self.encoder:
            x = layer(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar
    
    def encoder_forward_domain(self, x):
        if self.training:
            x = self.dropout(x)
        for layer in self.encoder_domain:
            x = layer(x)
        mu = self.mu_layer_domain(x)
        logvar = self.logvar_layer_domain(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decoder_forward(self, z):
        for layer in self.decoder:
            z = layer(z)
        return z
    
    def decoder_forward_domain(self, z):
        for layer in self.decoder_domain:
            z = layer(z)
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder_forward(x)
        mu_domain, logvar_domain = self.encoder_forward_domain(x)
        z = self.reparameterize(mu, logvar)
        z_domain = self.reparameterize(mu_domain, logvar_domain)
        domain_prediction = self.decoder_forward_domain(z_domain)
        recon_x = self.decoder_forward(z)
        
        return recon_x, mu, logvar, domain_prediction, mu_domain, logvar_domain