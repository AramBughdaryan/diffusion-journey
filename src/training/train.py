import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vae import VAE
from src.data.dataset import MNISTDataset
from src.training.trainer import Trainer
import torch


if __name__ == "__main__":
    dataset = MNISTDataset(data_path='data/mnist_data', train=True, use_zstd=False)
    # For the first time, preprocess the data and save it
    # dataset.preprocess_and_save(raw_data_path='raw_mnist', save_path='data/mnist_data', use_zstd=False)
    model = VAE(input_dim=28 * 28, latent_dim=8)
    trainer = Trainer(model=model, dataset=dataset, batch_size=128, learning_rate=0.0001, loss_fn_name='mae')
    # trainer._load_snapshot('checkpoints/model_epoch_100.pth')
    model.load_state_dict(
        torch.load('checkpoints/model_epoch_290.pth')
        )
    trainer.train(epochs=5000)
