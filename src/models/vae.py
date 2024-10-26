# import torch
# import torch.nn as nn

# class VAE(nn.Module):
#     def __init__(self, latent_dim=16):
#         super(VAE, self).__init__()
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=2),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
#             nn.Conv2d(64, 32, kernel_size=3, stride=2),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(),
#             nn.Flatten()
#         )
        
#         self.flattened_size = 128
        
#         # Linear layers for mean and log variance
#         self.mean_predictor = nn.Linear(self.flattened_size, latent_dim)
#         self.log_std_predictor = nn.Linear(self.flattened_size, latent_dim)
        
#         # Decoder
#         self.decoder_fc = nn.Linear(latent_dim, self.flattened_size * 3 * 3)
        
#         self.decoder = nn.Sequential(
#             # 4x4 -> 7x7
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1),
#             nn.ReLU(),
#             # 7x7 -> 14x14
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
#             nn.ReLU(),
#             # 14x14 -> 28x28
#             nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )

#     def reparametrization(self, mean, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def encode(self, x):
#         encoded_input = self.encoder(x)
#         # Adding shape debugging
#         encoded_mean = self.mean_predictor(encoded_input)
#         encoded_log_std = self.log_std_predictor(encoded_input)
#         return encoded_mean, encoded_log_std

#     def decode(self, z):
#         z = self.decoder_fc(z)
#         # Reshape back to spatial dimensions: batch_size x 128 x 4 x 4
#         z = z.view(-1, 128, 3, 3)
#         return self.decoder(z)

#     def forward(self, data):
#         mean, logvar = self.encode(data)
#         z = self.reparametrization(mean, logvar)
#         x_hat = self.decode(z)
#         return x_hat, mean, logvar

#     def generate(self, num_samples):
#         z = torch.randn(num_samples, self.mean_predictor.out_features)
#         if torch.cuda.is_available():
#             z = z.cuda()
#         out = self.decode(z)
#         return out

import torch
import torch.nn as nn


class VAE(torch.nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.mean_predictor = nn.Linear(in_features=latent_dim, out_features=4)
        self.log_std_predictor = nn.Linear(in_features=latent_dim, out_features=4)

        self.decoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def reparametrization(self, mean, log_var):
        z = torch.randn_like(log_var, device=log_var.device)
        std = torch.exp(0.5 * log_var)
        x = mean + z * std

        return x

    def encode(self, x):
        encoded_input = self.encoder(x)
        encoded_mean = self.mean_predictor(encoded_input)
        encoded_log_std = self.log_std_predictor(encoded_input)

        return encoded_mean, encoded_log_std

    def decode(self, input):
        return self.decoder(input)

    def forward(self, data):
        mean, logvar = self.encode(data)
        x = self.reparametrization(mean=mean, log_var=logvar)
        x_hat = self.decode(x)

        return x_hat, mean, logvar

    def generate(self, num_samples):
        x = torch.randn(num_samples, 4, device='cuda')
        out = self.decode(x)
        return out