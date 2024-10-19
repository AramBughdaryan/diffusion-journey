# Class trainer.
#  Must have _load_snapshot(to load pretrained model)
# _run_batch() Must run model ob batch of data. takes input and targets as input. does backward and opt.step in that function
# _run_epoch() Must run epoch 
# _save_snapshot save model state_dict 
# train function which must be called from train.py file.
# Must initialize dataloader in init function which takes dataset, batch_size etc.
# Implement loss function.
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


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
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.loss_fn_name = loss_fn_name

        self.dataloader = dataset.get_dataloader(batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # TODO: Implement loss function
        if loss_fn_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_fn_name == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}")

    def _run_batch(self, input, target):
        """
        Run model on a batch of data, compute loss, and update model weights.
        Args:
            input: Batch of input data.
            target: Corresponding target labels.
        """
        self.optimizer.zero_grad()
        output = self.model(input)
        
        if isinstance(output, tuple):
            output = output[0]

        loss = self.loss_fn(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def _run_epoch(self):
        """
        Run a single epoch through the dataset.
        """
        self.model.train()
        total_loss = 0

        for batch_idx, (input, target) in enumerate(self.dataloader):
            batch_size = input.size(0)
            input_flattened = input.view(batch_size, -1)
            batch_loss = self._run_batch(input_flattened, target)
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

    def train(self, epochs):
        """
        Train the model for a specified number of epochs.
        Args:
            epochs: Number of epochs to train the model for.
        """
        for epoch in range(1, epochs + 1):
            avg_loss = self._run_epoch()
            print(f'Epoch {epoch}, Loss: {avg_loss}')
            self._save_snapshot(epoch)
