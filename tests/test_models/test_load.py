"""!@file test_load.py

@brief This file contains the unit tests for the data loading

@details This file contains the unit tests for the data loading and model components of the lung segmentation pipeline.

@author Created by C. Grivot on 14/03/2024
"""
import unittest
import torch
from src.models.load_data import LungSegmentationDataset
from src.models.unet import SimpleUNet, train_model


class TestLungSegmentationComponents(unittest.TestCase):
    """! Class representing the unit tests for the data loading and model components
    of the lung segmentation pipeline."""

    def test_dataset_loading(self):
        """! Test the loading of the dataset to ensure it is correctly initialized and has the correct length."""
        # Test dataset initialization and length
        dataset = LungSegmentationDataset(
            image_dir="Dataset/processed", mask_dir="Dataset/Segmentations"
        )
        self.assertIsInstance(dataset, LungSegmentationDataset)
        self.assertTrue(len(dataset) > 0)

    def test_dataset_item(self):
        """! Test the retrieval of an item from the dataset to ensure it returns the correct shapes and types."""
        # Test retrieving an item from the dataset
        dataset = LungSegmentationDataset(
            image_dir="Dataset/processed", mask_dir="Dataset/Segmentations"
        )
        image, mask = dataset[0]  # Retrieve the first sample
        print("Image shape:", image.shape)
        print("Mask shape:", mask.shape)
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(image.shape[0], 1)  # Channel dimension should be 1
        self.assertEqual(mask.shape[0], 1)  # Channel dimension should be 1

    def test_model_instantiation(self):
        """! Test the instantiation of the model to ensure it is correctly initialized
        and has the correct output shape."""
        # Test model instantiation
        model = SimpleUNet(in_channels=1, out_channels=1)
        self.assertIsInstance(model, SimpleUNet)
        x = torch.randn(1, 1, 256, 256)  # Example input tensor
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape[1], 1)  # Output channel dimension

    def test_training_function_exists(self):
        """! Test the existence of the training function to ensure it is callable."""
        # Test if the training function is callable
        self.assertTrue(callable(train_model))


if __name__ == "__main__":
    unittest.main()
