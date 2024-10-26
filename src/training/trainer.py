# Class trainer.
#  Must have _load_snapshot(to load pretrained model)
# _run_batch() Must run model ob batch of data. takes input and targets as input. does backward and opt.step in that function
# _run_epoch() Must run epoch 
# _save_snapshot save model state_dict 
# train function which must be called from train.py file.
# Must initialize dataloader in init function which takes dataset, batch_size etc.
# Implement loss function.
import os
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: torch.nn.Module, dataset: Dataset, batch_size: int, learning_rate: float, loss_fn_name='cross_entropy', save_dir='checkpoints'):
        """
        Args:
            model: The PyTorch model to be trained.
            dataset: The dataset used for training.
            batch_size: The size of batches during training.
            learning_rate: The learning rate for the optimizer.
            loss_fn_name: Loss function to use ('cross_entropy', 'mse').
            save_dir: Directory to save model snapshots.
        """
        self.model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.loss_fn_name = loss_fn_name

        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


        if not os.path.exists('reconstructed_images'):
            os.makedirs('reconstructed_images')

        if not os.path.exists('generated_images'):
            os.makedirs('generated_images')

        if loss_fn_name == 'mse':
            self.reconstruction_loss_fn = nn.MSELoss()
        elif loss_fn_name == 'mae':
            self.reconstruction_loss_fn = nn.L1Loss()
        elif loss_fn_name == 'cross_entropy':
            self.reconstruction_loss_fn = nn.CrossEntropyLoss()
        else:

            raise ValueError(f"Unknown loss function: {loss_fn_name}")

    def _loss_fn(self, x_hat, mean, logvar, x):
        reconstruction_loss = self.reconstruction_loss_fn(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        logger.info(f'Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                    f'KL Divergence: {kl_divergence.item():.4f}, ')

        return reconstruction_loss + kl_divergence

    def _run_batch(self, input_batch, epoch, target=None):
        """
        Run model on a batch of data, compute loss, and update model weights.
        Args:
            input: Batch of input data.
            target: Corresponding target labels.
        """
        self.optimizer.zero_grad()
        x_hat, mean, logvar = self.model(input_batch)

        loss = self._loss_fn(
            x_hat=x_hat, mean=mean,
            logvar=logvar, x=input_batch
            )
        if torch.rand(1) > 0.9:
            img = x_hat[0].detach().cpu().numpy().reshape(28, 28)
            original_image = input_batch[0].detach().cpu().numpy().reshape(28, 28)
            plt.imsave(f'reconstructed_images/pred_image_epoch_{epoch}.png', img, cmap='gray')
            plt.imsave(f'reconstructed_images/original_image_epoch_{epoch}.png', original_image, cmap='gray')

        loss.backward()
        self.optimizer.step()

        return loss.item()
        

    def _run_epoch(self, epoch):
        """
        Run a single epoch through the dataset.
        """
        self.model.train()
        total_loss = 0

        for batch_idx, (input_batch, target) in enumerate(self.dataloader):
            batch_size = input_batch.size(0)
            # input_batch = torch.reshape(input_batch, (batch_size, 1, 28, 28))
            batch_loss = self._run_batch(input_batch, epoch)
            total_loss += batch_loss

        avg_loss = total_loss / len(self.dataloader)
        return avg_loss

    def _save_snapshot(self, epoch):
        """
        Save the model's state_dict (snapshot) to disk.
        Args:
            epoch: The current epoch number.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f'model_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f"Model snapshot saved to {save_path}")

    def _load_snapshot(self, checkpoint_path):
        """
        Load the model's state_dict (snapshot) from disk.
        Args:
            checkpoint_path: Path to the saved model checkpoint.
        """
        if os.path.isfile(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    def generate_images(self, epoch, num_samples=2):
        with torch.no_grad():
            generated_images = self.model.generate(num_samples)
            generated_images_np = generated_images.cpu().numpy()
            for i in range(num_samples):
                img = generated_images_np[i].reshape(28, 28)
                plt.imsave(f'generated_images/generated_epoch_{epoch}_img_{i}.png', img, cmap='gray')


    def train(self, epochs):
        """
        Train the model for a specified number of epochs.
        Args:
            epochs: Number of epochs to train the model for.
        """
        for epoch in range(1, epochs + 1):
            avg_loss = self._run_epoch(epoch=epoch)
            print(f'Epoch {epoch}, Loss: {avg_loss}')

            if epoch % 10 == 0:
                self._save_snapshot(epoch)
                self.generate_images(epoch=epoch, num_samples=2)
                
                        


