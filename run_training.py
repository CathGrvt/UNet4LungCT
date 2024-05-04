"""@file run_training.py

@brief Run training

@details This script runs the training of a model for lung segmentation.
The script loads the dataset, splits it into training and validation sets,
sets up the model, criterion, optimizer, and learning rate scheduler, and trains the model.
The script also includes a main function that is called when the script is run,
which loads the dataset, splits it into training and validation sets, sets up the model,
criterion, optimizer, and learning rate scheduler, and trains the model.
The main function also includes a try-except block to catch any exceptions
that occur during the training and log them before exiting.

@author Created by C. Grivot on 14/03/2024
"""
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging
from src.models.load_data import LungSegmentationDataset
from src.models.unet import SimpleUNet, train_model, CustomLoss
from src.utilities.metrics import visualize_training_performance

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    try:
        # Parameters for training
        mask_dir = "Dataset/Segmentations"
        batch_size = 3
        lr = 0.1
        num_epochs = 10
        in_channels = 1
        out_channels = 1

        # Dataset and DataLoader
        logging.info("Loading dataset...")
        dataset = LungSegmentationDataset(
            image_dir="Dataset/processed", mask_dir=mask_dir
        )

        # Split dataset
        logging.info("Splitting dataset...")
        train_dataset, val_dataset = train_test_split(
            dataset, test_size=0.33, random_state=42
        )
        image_datasets = {"train": train_dataset, "val": val_dataset}
        torch.save(image_datasets, "image_datasets.pth")

        dataloaders = {
            x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
            for x in ["train", "val"]
        }
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

        # Model setup
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        model = SimpleUNet(in_channels, out_channels).to(device)
        criterion = CustomLoss(weights=[0.5, 0.5])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Decay LR by a factor of 0.1 every 3 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Training
        logging.info("Starting training...")
        model = train_model(
            model,
            criterion,
            optimizer,
            exp_lr_scheduler,
            dataloaders,
            device,
            dataset_sizes,
            num_epochs=num_epochs,
        )

        # Performance Visualisation
        visualize_training_performance("model_checkpoint.pth", device)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
