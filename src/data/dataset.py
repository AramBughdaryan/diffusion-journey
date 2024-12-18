# write a MNISTDataset class which inherits from pytorch dataset. 
# Must be Implemented __getitem__(idx) function which returns 
# image in processed_format(in torch.tensor format and values in [0,1] range)
# Also would be great if we could process raw data into let's say NPZ(np.savez())/zstandard 
# format and then use it while training
import io

import torch
import numpy as np
import zstandard as zstd
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, data_path: str, train=True, transform=None, use_zstd=False):
        """
        Args:
            data_path (str): Path to preprocessed npz/zstd data file.
            train (bool): If True, load training data; otherwise, load test data.
            transform (callable, optional): Optional transform to apply on a sample.
            use_zstd (bool): If True, use zstandard compressed data.
        """
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.use_zstd = use_zstd
        self._load_data()
        self.images = torch.tensor(self.images).to('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_data(self):
        """Load the data from NPZ or Zstandard file."""
        if self.use_zstd:
            with open(f"{self.data_path}.zst", 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(f.read())
                npz_data = np.load(decompressed)
        else:
            npz_data = np.load(f"{self.data_path}.npz")
        
        if self.train:
            self.images = npz_data['train_images']
            self.labels = npz_data['train_labels']
        else:
            self.images = npz_data['test_images']
            self.labels = npz_data['test_labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item at index idx, returning image as a torch.Tensor and corresponding label."""
        image = self.images[idx]
        label = self.labels[idx]
        image = image.reshape(-1)
        return image, label
    
    # def get_dataloader(self, batch_size: int):
    #     return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=True)
        

    @staticmethod
    def preprocess_and_save(raw_data_path: str, save_path: str, use_zstd=False):
        """
        Preprocess the raw MNIST dataset and save it as NPZ or Zstandard file.
        
        Args:
            raw_data_path (str): Path to store the raw MNIST data temporarily.
            save_path (str): Path to save the preprocessed NPZ/Zstandard data.
            use_zstd (bool): If True, compress with zstandard.
        """
        # SSL Certificate issue workaround
        # import ssl
        # ssl._create_default_https_context = ssl._create_unverified_context
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_data = datasets.MNIST(root=raw_data_path, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root=raw_data_path, train=False, download=True, transform=transform)
        
        train_images = np.stack([train_data[i][0].numpy().squeeze() for i in range(len(train_data))])
        train_labels = np.array([train_data[i][1] for i in range(len(train_data))])

        test_images = np.stack([test_data[i][0].numpy().squeeze() for i in range(len(test_data))])
        test_labels = np.array([test_data[i][1] for i in range(len(test_data))])
            
        if use_zstd:
                buffer = io.BytesIO()
                np.savez(buffer, 
                        train_images=train_images, 
                        train_labels=train_labels, 
                        test_images=test_images, 
                        test_labels=test_labels)
                buffer.seek(0)

                compressed_data = zstd.ZstdCompressor().compress(buffer.read())

                with open(f"{save_path}.zst", 'wb') as f:
                    f.write(compressed_data)
        else:
            np.savez_compressed(f"{save_path}.npz", 
                                train_images=train_images, 
                                train_labels=train_labels, 
                                test_images=test_images, 
                                test_labels=test_labels)
