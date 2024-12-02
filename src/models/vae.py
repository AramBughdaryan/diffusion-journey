import torch
import torch.nn as nn


class VAE(torch.nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latend_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.mean_predictor = nn.Linear(in_features=128, out_features=latent_dim)
        self.log_std_predictor = nn.Linear(in_features=128, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def reparametrization(self, mean, log_var):
        z = torch.randn_like(log_var, device=log_var.device)
        std = torch.exp(0.5 * log_var)
        x = mean + z * std

        return x

    def encode(self, x):
        encoded_input = self.encoder(x.view(-1, 28 * 28))
        encoded_mean = self.mean_predictor(encoded_input)
        encoded_log_std = self.log_std_predictor(encoded_input)

        return encoded_mean, encoded_log_std

    def decode(self, input_batch):
        return self.decoder(input_batch)

    def forward(self, data):
        mean, logvar = self.encode(data)
        x = self.reparametrization(mean=mean, log_var=logvar)
        x_hat = self.decode(x)

        return x_hat, mean, logvar

    def generate(self, num_samples):
        x = torch.randn(num_samples, 4, device = 'cuda' if torch.cuda.is_available() else 'cpu')
        out = self.decode(x)
        return out