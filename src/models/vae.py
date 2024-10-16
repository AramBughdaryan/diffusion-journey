import torch
import torch.nn as nn


class VAE(torch.nn.Module):
    def __init__(self, input_dim=64*64, latent_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.mean_predictor = nn.Linear(in_features=latent_dim, out_features=4)
        self.std_predictor = nn.Linear(in_features=latent_dim, out_features=4)
        self.decoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def reparametrization(self, mean, var):
        z = torch.randn_like(var, device=var.device)
        x = mean + z * var

        return x

    def encode(self, x):
        encoded_input = self.encoder(x)
        encoded_mean = self.mean_predictor(encoded_input)
        encoded_log_std = self.std_predictor(encoded_input)

        return encoded_mean, encoded_log_std

    def decode(self, input):
        return self.decoder(input)

    def forward(self, data):
        mean, logvar = self.encode(data)
        x = self.reparametrization(mean=mean, var=logvar)
        x_hat = self.decode(x)

        return x_hat, mean, logvar
