"""!@file load_data.py

@brief Load data from the dataset

@details This function loads the data from the dataset and returns the data in the form of a PyTorch Dataset object. The dataset is loaded from the specified image and mask directories, and the data is returned as a PyTorch Dataset object. The dataset is used to train the model in the lung segmentation pipeline.

@author Created by C. Grivot on 14/03/2024
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class LungSegmentationDataset(Dataset):
    """! Class for loading data from the dataset.
    """
    def __init__(self, image_dir, mask_dir):
        """! Initialize the dataset by loading the data from the specified directories.
        @param image_dir The directory containing the input images.
        @param mask_dir The directory containing the target masks."""
        self.X_slices = []
        self.y_slices = []
        for filename in os.listdir(image_dir):
            if not filename.startswith("."):
                X_array = np.load(os.path.join(image_dir, filename))
                for i in range(X_array.shape[0]):
                    self.X_slices.append(X_array[i])

        for filename in os.listdir(mask_dir):
            if not filename.startswith("."):
                y_array = np.load(os.path.join(mask_dir, filename))["masks"]
                for i in range(y_array.shape[0]):
                    self.y_slices.append(y_array[i])

    def __len__(self):
        """! Return the length of the dataset."""
        return len(self.X_slices)

    def __getitem__(self, idx):
        """! Retrieve an item from the dataset.
        @param idx The index of the item to retrieve.
        """
        X = torch.from_numpy(self.X_slices[idx].astype(np.float32))
        y = torch.from_numpy(self.y_slices[idx].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        return X, y
