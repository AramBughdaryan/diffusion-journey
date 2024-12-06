import torch
import numpy as np

class DDPMSampler:
    # Scheduler as in the picture.
    def __init__(self, generator: torch.Generator, num_training_steps=1000,
                beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpa_cumprod = torch.cumprod(self, self.alphas, 0) # [alpha_0, alpha_0 * alpha1, alpha_0 * alpha1 * alpha_2, ...]
        self.one = torch.tensor(1.0)
        
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1]).copy()
    
    def set_inference_timesteps(self, num_inference_timesteps=50):
        self.num_inference_steps = num_inference_timesteps
        # 999, 998, 997, ..., 0 = 1000 steps
        # We want to have timesteps like the line below
        # 999, 999 - 20, ..., 20, 0 = 50 steps
        
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor):
        # p(x_t | x_0) = N(x_t; sqrt(alpha_t) * x_0, (1 - alpha_t) * I)
        alpha_cumprod = self.alpa_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timestep.to(original_samples.device)
        
        sqrt_alpha_prod = alpha_cumprod[timestep] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timestep]) ** 0.5 # standard deviation
        # TODO. understand why we unsqeeze or flatten
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqeeze(-1)
        
        # According to the equation (4) of the DDPM paper.
        noise = torch.rand(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + (sqrt_one_minus_alpha_prod) * noise
        
        return noisy_samples
        
        
        
        
        