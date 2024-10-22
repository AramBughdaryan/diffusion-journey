import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE
from data.dataset import MNISTDataset
from training.trainer import Trainer


if __name__ == "__main__":
    dataset = MNISTDataset(data_path='data/mnist_data', train=True, use_zstd=False)
    # For the first time, preprocess the data and save it
    # dataset.preprocess_and_save(raw_data_path='raw_mnist', save_path='data/mnist_data', use_zstd=False)
    model = VAE(input_dim=28*28, latent_dim=4)
    trainer = Trainer(model=model, dataset=dataset, batch_size=32, learning_rate=0.001, loss_fn_name='cross_entropy')
    # trainer._load_snapshot('checkpoints/model_epoch_30.pth')
    trainer.train(epochs=50)
