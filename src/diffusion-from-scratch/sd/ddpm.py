import torch
import numpy as np

class DDPMSampler:
    # Scheduler as in the picture.
    def __init__(self, generator: torch.Generator, num_training_steps=1000,
                beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alpa_cumprod = torch.cumprod(self.alphas, 0) # [alpha_0, alpha_0 * alpha1, alpha_0 * alpha1 * alpha_2, ...]
        self.one = torch.tensor(1.0)
        
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, num_inference_timesteps=50):
        self.num_inference_steps = num_inference_timesteps
        # 999, 998, 997, ..., 0 = 1000 steps
        # We want to have timesteps like the line below
        # 999, 999 - 20, ..., 20, 0 = 50 steps
        
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def _get_previous_timestep(self, timestep: int):
        # during inference if we are at step 980 we need previous timestep 
        # which is 980 - 20 in case we don 50 inference iterations
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def _get_variance(self, timesteps: int) -> torch.Tensor:
        previous_timestep = self._get_previous_timestep(timesteps)
        alpha_prod_t = self.alpa_cumprod[timesteps]
        alpha_prod_t_prev = self.alpa_cumprod[previous_timestep] if previous_timestep >= 0 else self.one
        current_betta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        #  Which is the same as: betta_t = self.betas[timesteps]
        # Computed using formula (7) from ddpm paper
        variance = (1 - alpha_prod_t_prev) * current_betta_t / (1 - alpha_prod_t)
        variance = variance.clamp(min=1e-20)
        return variance
        
    def set_strength(self, strength):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
    
    
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        # using equation (11) from ddpm paper.
        # x_(t-1) = 1 / sqrt(alpha_t) ( x_t - betta_t / (1 - alpha_t_tilda) * predicted_noise) + sigma_t * z, where z ~ N(0, I)
        # or which is equivalent to equation (7)
        #  q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu^tilda_t(x_t, x_0, betta^tilda_t * I))
        # mu_tilda_t(x_t, x_0) = (sqrt(alpha_prod_{t-1}) * betta_t * x_0) / (1 - alpha^tilda_t) 
        # + sqrt(alpha_t)(1 - alpha^tilda_{t-1} / (1 - alpha^tilda_t) * x_t)
        # where betta^tilda_t = (1 - alpha^tilda_{t-1} / (1 - alpha^tilda_t) * betta_t)
        # and x_0 predicted is (x_t - sqrt(1 - alpha^tilda_t * predicted_noise_at_x_t)) / sqrt(alpha^tilda_t)

        # model_output: predicted noise (epsilon_theta)
        t = timestep
        prev_t = self._get_previous_timestep(t)
        
        alpha_prod_t = self.alpa_cumprod[timestep]
        alpha_prod_t_prev = self.alpa_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # compute the predicted original sample using formula (15) from ddpm
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output ) / alpha_prod_t ** 0.5
        
        # Compute the coefficients for pred_original_sample and current sample x_t
        predicted_original_sample_coeffs = (alpha_prod_t_prev ** 0.5 * current_beta_t) / (beta_prod_t)
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        
        # Compute the predicted previous sample mean
        pred_prev_sample = predicted_original_sample_coeffs * pred_original_sample + current_sample_coeff * latents
        
        variance = 0
        if t > 0:
            device = model_output.device
            
            noise = torch.randn(model_output.shape, 
                            generator=self.generator, device=device
                            )
            variance = (self._get_variance(timestep) ** 0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample

    
    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor):
        # p(x_t | x_0) = N(x_t; sqrt(alpha_t_cumprod) * x_0, (1 - alpha_t_cumprod) * I)
        alpha_cumprod = self.alpa_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timestep = timestep.to(original_samples.device)
        
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

        
        
        
        
        