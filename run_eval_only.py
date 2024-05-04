"""!@file run_eval_only.py
@brief This script runs the evaluation of a trained model and visualizes its performance.

@details The script loads a trained model, performs predictions on chosen slices and random test cases,
evaluates the model using dice scores and IoU scores, and plots the metrics.
The script also includes a main function that is called when the script is run,
which loads a trained model, performs predictions on chosen slices and random test cases,
evaluates the model using dice scores and IoU scores, and plots the metrics.
The main function also includes a try-except block to catch any exceptions
that occur during the evaluation and log them before exiting.

@author Created by C. Grivot on 14/03/2024
"""
import torch
from torch.utils.data import DataLoader
from src.models.unet import SimpleUNet
from src.models.load_data import LungSegmentationDataset
from src.utilities.metrics import (
    evaluate_model,
    plot_metrics,
    # predict_random_test_case,
    # visualize_slice_prediction,
    visualize_examples,
)
import logging
import os
from sklearn.model_selection import train_test_split

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """! This function is the entry point of the script. It loads a trained model, performs predictions on chosen slices
    and random test cases, evaluates the model using dice scores and IoU scores, and plots the metrics.
    """
    try:
        # Setup device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Model loading
        model_checkpoint_path = "model_checkpoint.pth"
        if not os.path.exists(model_checkpoint_path):
            logging.error(f"Model checkpoint file not found: {model_checkpoint_path}")
            return

        model = SimpleUNet(1, 1).to(device)
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        weights = checkpoint["model_state_dict"]
        model.load_state_dict(weights)

        model.eval()  # Set model to evaluation mode

        # Visualizations and Predictions
        logging.info("All visualisations are saved not shown!")

        # logging.info("Visualising chosen slice...")
        # visualize_slice_prediction(
        #     model,
        #     device,
        #     "Dataset/processed/Case_010.npy",
        #     "Dataset/Segmentations/Case_010_seg.npz",
        #     42,
        # )

        # # Evaluate the model
        logging.info("Loading dataset...")
        # image_datasets = torch.load("image_datasets.pth", map_location=device)
        # OR
        dataset = LungSegmentationDataset(
            image_dir="Dataset/processed", mask_dir="Dataset/Segmentations"
        )
        train_dataset, val_dataset = train_test_split(
            dataset, test_size=0.33, random_state=42
        )
        image_datasets = {"train": train_dataset, "val": val_dataset}
        image_datasets = {"train": train_dataset, "val": val_dataset}
        dataloader_val = DataLoader(image_datasets["val"], batch_size=1, shuffle=True)
        dataloader_train = DataLoader(
            image_datasets["train"], batch_size=1, shuffle=False
        )

        logging.info("Evaluating model and calculating scores...")
        dice_scores_test, iou_scores_test, accuracy_scores_test = evaluate_model(
            model, dataloader_val
        )
        dice_scores_train, iou_scores_train, accuracy_scores_train = evaluate_model(
            model, dataloader_train
        )

        logging.info("Plotting metrics...")
        # Plot metrics
        plot_metrics(dice_scores_train, dice_scores_test)
        plot_metrics(
            iou_scores_train, iou_scores_test, title="IoU scores", ylabel="IoU"
        )
        plot_metrics(
            accuracy_scores_train,
            accuracy_scores_test,
            title="Accuracy scores",
            ylabel="Accuracy",
        )

        logging.info("Visualising examples...")
        visualize_examples(dataloader_val, model, device)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
