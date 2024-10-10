import torch
import torch.nn as nn


class VAE(torch.nn.Module):
    def __init__(self, latent_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = VAEEncoder()
        self.mean_predictor = nn.Linear(in_features=latent_dim, out_features=2)
        self.std_predictor = nn.Linear(in_features=latent_dim, out_features=2) # why to use log_variation?
        # TODO handle forward pass? need to do training such way so that
        # then we can sample from it using Z from standard normal and give input to our decoder mean + std * Z
        self.decoder = VAEDecoder()

    def forward(self, data):
        encoded_input = self.encoder(data)
        encoded_mean = self.mean_predictor(encoded_input)
        encoded_std = self.std_predictor(encoded_input)


class VAEDecoder(torch.nn.Module):
    pass

class VAEEncoder(torch.nn.Module):
    pass
