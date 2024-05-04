"""!@file run_pipeline.py

@brief This script runs the entire pipeline, from data conversion to model evaluation and visualization.

@details The script loads the configuration file, validates the dataset path,
converts the dataset to numpy format, trains the model, evaluates the model,
and visualizes the model performance and predictions.
The script also includes a main function that is called when the script is run,
which loads the configuration file, validates the dataset path, converts the dataset to numpy format,
trains the model, evaluates the model, and visualizes the model performance and predictions.
The main function also includes a try-except block to catch any exceptions
that occur during the pipeline and log them before exiting.

@author Created by C. Grivot on 14/03/2024
"""
import json
import logging
import os
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.models.unet import SimpleUNet, train_model, CustomLoss
from src.models.load_data import LungSegmentationDataset
from src.utilities.metrics import (
    evaluate_model,
    plot_metrics,
    # predict_random_test_case,
    visualize_slice_prediction,
    visualize_examples,
    visualize_training_performance,
)
from src.data_processing.dicom_to_numpy import convert_to_np

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path):
    """! Load the JSON configuration file."""
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def validate_dataset_path(path):
    """! Validate the dataset path."""
    if not os.path.exists(path):
        logging.error(f"Dataset path does not exist: {path}")
        sys.exit(1)
    if not os.listdir(path):
        logging.error(f"Dataset directory is empty: {path}")
        sys.exit(1)
    logging.info(f"Dataset path validated: {path}")


def main(config_path):
    config = load_config(config_path)
    dataset_path = config["dataset_path"]
    try:
        # Data Processing
        validate_dataset_path(dataset_path)
        logging.info("Starting data conversion")
        convert_to_np(dataset_path)
        logging.info("Data conversion completed successfully.")

        # Model Training Preparation
        mask_dir = config["mask_dir"]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        dataset = LungSegmentationDataset(
            image_dir="Dataset/processed", mask_dir=mask_dir
        )
        train_dataset, val_dataset = train_test_split(
            dataset,
            test_size=config["test_split_ratio"],
            random_state=config["random_state"],
        )
        dataloaders = {
            x: DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
            for x, dataset in [("train", train_dataset), ("val", val_dataset)]
        }
        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

        # Model Training
        model = SimpleUNet(config["in_channels"], config["out_channels"]).to(device)
        criterion = CustomLoss(weights=[0.5, 0.5])
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        model = train_model(
            model,
            criterion,
            optimizer,
            exp_lr_scheduler,
            dataloaders,
            device,
            dataset_sizes,
            num_epochs=config["num_epochs"],
        )

        # Performance Visualisation
        visualize_training_performance(config["model_checkpoint_path"], device)

        # Evaluation and Visualisation
        logging.info("Evaluating model and visualising results...")

        dice_scores_test, iou_scores_test, accuracy_scores_test = evaluate_model(
            model,
            dataloaders["val"],  # with batch_size=3, modify it to 1 if needed
        )
        dice_scores_train, iou_scores_train, accuracy_scores_train = evaluate_model(
            model, dataloaders["train"]  # with batch_size=3
        )

        plot_metrics(dice_scores_train, dice_scores_test, "DSC", "Dice Score")
        plot_metrics(iou_scores_train, iou_scores_test, "IoU", "IoU Score")
        plot_metrics(
            accuracy_scores_train, accuracy_scores_test, "Accuracy", "Accuracy Score"
        )

        # Best, in-between and worst slices
        visualize_examples(dataloaders["val"], model, device)
        # Choose a slice to visualise
        visualize_slice_prediction(
            model,
            device,
            "Dataset/processed/Case_010.npy",
            "Dataset/Segmentations/Case_010_seg.npz",
            42,
        )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(config_path)
