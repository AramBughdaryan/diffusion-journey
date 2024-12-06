import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torchvision import datasets, transforms

from src.models.vae import VAE
from src.data.dataset import MNISTDataset
from src.training.trainer import Trainer



if __name__ == "__main__":
    # dataset = MNISTDataset(data_path='data/mnist_data', train=True, use_zstd=False)
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # For the first time, preprocess the data and save it
    # dataset.preprocess_and_save(raw_data_path='raw_mnist', save_path='data/mnist_data', use_zstd=False)
    model = VAE(input_dim=28 * 28, latent_dim=4)
    trainer = Trainer(model=model, dataloader=train_loader, batch_size=128, learning_rate=0.001, loss_fn_name='cross_entropy')
    # trainer._load_snapshot('checkpoints/model_epoch_100.pth')
    model.load_state_dict(
        torch.load('checkpoints/model_epoch_100.pth')
        )
    trainer.train(epochs=5000)
