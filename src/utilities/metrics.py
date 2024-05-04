"""!@file metrics.py

@brief Script for calculating and visualizing evaluation metrics for the lung segmentation pipeline.

@details This script contains functions for calculating and visualizing evaluation metrics
for the lung segmentation pipeline, including the Dice Similarity Coefficient (DSC),
Intersection over Union (IoU), and binary accuracy.
The functions are used to evaluate the performance of the model, visualize examples of 2D slices,
and plot the training performance.

@author Created by C. Grivot on 14/03/2024
"""
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from torchmetrics.classification import BinaryAccuracy


def visualize_examples(loader, model, device):
    """! Visualize examples of 2D slices with best, worst, and in-between DSC values.

    @param loader: The DataLoader containing the dataset to visualize.
    @param model: The trained model used for making predictions.
    @param device: The device (CPU or GPU) to perform computations on.
    """
    model.to(device)
    model.eval()
    dsc_scores = []
    images, truths, preds = [], [], []

    with torch.no_grad():
        for image, truth in loader:
            image, truth = image.to(device), truth.to(device)
            pred = model(image)
            dsc = dice_coefficient(pred, truth)
            dsc_scores.append(dsc.item())
            images.append(image.cpu())
            truths.append(truth.cpu())
            preds.append(pred.cpu())

    # Sort slices based on DSC scores and select examples
    sorted_indices = sorted(range(len(dsc_scores)), key=lambda k: dsc_scores[k])
    best_indices = sorted_indices[-3:]
    worst_indices = sorted_indices[:3]
    mid_point = len(sorted_indices) // 2
    in_between_indices = sorted_indices[mid_point - 1 : mid_point + 2]

    # Function to plot a single column
    def plot_column(index, title, category):
        """! Plot column of 3 images comparing the orignal image, the ground truth and the prediction
        @param index: The index of the column to plot.
        @param title: The title of the column
        @param category: The category of the column (best, worst, or in-between).
        """
        plt.subplot(3, 3, index)
        image = images[category][0][0, :, :]
        plt.imshow(image, cmap="gray")
        plt.title(f"{title} Image")
        plt.axis("off")

        plt.subplot(3, 3, index + 1)
        mask = truths[category][0][0, :, :]
        plt.imshow(mask, cmap="jet")
        plt.title(f"{title} Ground Truth")
        plt.axis("off")

        plt.subplot(3, 3, index + 2)
        pred = torch.sigmoid(preds[category] > 0.5)
        pred = pred[0, 0, :, :]
        plt.imshow(pred, cmap="jet")
        plt.title(f"{title} Prediction")
        plt.axis("off")
        plt.savefig(f"plots/{title}Examples_seg_scores.png")

    # Create figure
    plt.figure(figsize=(9, 9))

    for i, idx in enumerate(best_indices):
        plot_column(i * 3 + 1, "Best", idx)

    for i, idx in enumerate(worst_indices):
        plot_column(i * 3 + 1, "Worst", idx)

    for i, idx in enumerate(in_between_indices):
        plot_column(i * 3 + 1, "In-between", idx)

    plt.tight_layout()


def evaluate_model(model, test_loader):
    """!Evaluate the model using Dice scores, IoU scores, and binary accuracy.
    @param model: The trained model used for making predictions.
    @param test_loader: The DataLoader containing the test dataset.

    @return: A tuple containing lists of Dice scores, IoU scores, and accuracy scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    iou_scores = []  # Initialize list to store IoU and Dice scores
    dice_scores = []
    accuracy_metric = BinaryAccuracy(threshold=0.5).to(device)
    accuracy_scores = []  # List to store accuracy for each 2D slice
    with torch.no_grad():  # No need to track gradients
        for inputs, true_masks in test_loader:
            inputs, true_masks = inputs.to(device), true_masks.to(device)
            predictions = model(inputs)
            dice_score = dice_coefficient(predictions, true_masks)
            dice_scores.append(dice_score.item())
            iou_score = compute_iou(predictions, true_masks)
            iou_scores.append(iou_score.item())
            # Compute binary accuracy for each 2D slice
            batch_accuracy_scores = []
            for pred_slice, true_slice in zip(predictions, true_masks):
                # Squeeze the slices to remove the channel dimension
                pred_slice = torch.sigmoid(pred_slice).squeeze()
                true_slice = true_slice.squeeze()
                # BinaryAccuracy expects inputs of shape (B, ...) or (...,). Since we're
                # evaluating slice-by-slice, we add a batch dimension of 1 to comply.
                slice_accuracy = accuracy_metric(
                    pred_slice.unsqueeze(0), true_slice.unsqueeze(0)
                )
                batch_accuracy_scores.append(slice_accuracy.item())

            accuracy_scores.extend(batch_accuracy_scores)

    # Reset the metric for future use
    accuracy_metric.reset()

    return dice_scores, iou_scores, accuracy_scores


def compute_iou(prediction_logits, ground_truth):
    """!Compute IoU between prediction logits and ground truth masks.
    Converts logits to probabilities using sigmoid, then thresholds to get binary mask.

    @param prediction_logits: The model's prediction logits.
    @param ground_truth: The ground truth masks.
    """
    prediction_probs = torch.sigmoid(
        prediction_logits
    )  # Convert logits to probabilities
    prediction_binary = (
        prediction_probs > 0.5
    )  # Apply threshold to get binary predictions
    prediction_binary = prediction_binary.to(
        torch.bool
    )  # Ensure binary predictions are in boolean format

    intersection = (
        (prediction_binary & ground_truth.bool()).float().sum()
    )  # Compute intersection
    union = (prediction_binary | ground_truth.bool()).float().sum()  # Compute union
    iou = intersection / union  # Calculate IoU
    iou = iou.mean()  # Average over the batch
    return iou


def visualize_training_performance(checkpoint_path, device):
    """!Load a model checkpoint and visualize the training loss and accuracy curves.

    Parameters:
    @param checkpoint_path: The file path to the model checkpoint.
    @param device: The device (CPU or GPU) to which the model is mapped.
    """
    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    train_losses = checkpoint["losses"]
    train_acc = checkpoint["accuracies"]

    # Plot training loss and accuracy
    plt.figure(figsize=(14, 6))

    # Training loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    # Training accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Accuracy", color="orange", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/Train_loss_acc.png")
    plt.show()


def visualize_slice_prediction(
    model, device, image_file_path, mask_file_path, slice_id, threshold=0.5
):
    """!Visualize a specific slice from a dataset, displaying the original image, its ground truth mask,
    and the model's prediction.

    Parameters:
    @param model: The trained model used for making predictions.
    @param device: The device (CPU or GPU) to perform computations on.
    @param image_file_path: Path to the NumPy file containing the image data.
    @param mask_file_path: Path to the NumPy file containing the mask data.
    @param slice_id: The ID of the slice to be visualized and predicted.
    @param threshold: Threshold for converting model outputs to binary predictions. Defaults to 0.5.
    """
    # Load the image and mask arrays
    mask_array = np.load(mask_file_path)["masks"]
    image_array = np.load(image_file_path)

    # Convert arrays to tensors
    image_tensor = torch.Tensor(image_array)
    mask_tensor = torch.Tensor(mask_array)

    # Select the specified slice
    curr_image = image_tensor[slice_id, :, :]
    curr_mask = mask_tensor[slice_id, :, :]

    # Visualise the image and mask
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(curr_image, cmap="gray")
    plt.title("Image of the chosen slice", wrap=True)
    plt.subplot(1, 3, 2)
    plt.imshow(curr_mask, cmap="jet")
    plt.title("Ground truth mask of the chosen slice", wrap=True)

    # Prepare the image for prediction
    curr_image = curr_image.unsqueeze(0).unsqueeze(0).to(device)

    # Make prediction
    curr_pred = model(curr_image)
    curr_pred = torch.sigmoid(curr_pred)
    curr_pred = (curr_pred > threshold).float().cpu()[0, 0, :, :]

    # Display the prediction
    plt.subplot(1, 3, 3)
    plt.imshow(curr_pred, cmap="jet")
    plt.title("Model prediction of the chosen slice", wrap=True)
    plt.tight_layout()
    plt.savefig("plots/example_pred_chosen_slice.png")


def plot_metrics(
    train_dsc, test_dsc, title="Dice Similarity Coefficient", ylabel="DSC"
):
    """!Plot the Dice Similarity Coefficient (DSC) scores and also other scores for the training and test sets.

    @param train_dsc: The list of scores for the training set.
    @param test_dsc: The list of scores for the test set.
    @param title: The title of the plot. Defaults to "Dice Similarity Coefficient".
    @param ylabel: The label for the y-axis. Defaults to "DSC".
    """
    plt.figure(figsize=(12, 6))

    # Training set plot
    plt.subplot(1, 2, 1)
    plt.scatter(np.arange(len(train_dsc)), train_dsc, label="Train scores")
    avg_train_dsc = sum(train_dsc) / len(train_dsc)  # Calculate average train DSC
    plt.axhline(
        y=avg_train_dsc,
        color="r",
        linestyle="--",
        label=f"Average: {avg_train_dsc:.3f}",
    )
    plt.title(f"{title} - Training Set")
    plt.xlabel("Slice")
    plt.ylabel(ylabel)
    plt.legend()

    # Test set plot
    plt.subplot(1, 2, 2)
    plt.scatter(np.arange(len(test_dsc)), test_dsc, label="Test scores")
    avg_test_dsc = sum(test_dsc) / len(test_dsc)  # Calculate average test DSC
    plt.axhline(
        y=avg_test_dsc, color="r", linestyle="--", label=f"Average: {avg_test_dsc:.3f}"
    )
    plt.title(f"{title} - Test Set")
    plt.xlabel("Slice")
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/model_scores_{title}.png")
    # plt.show()  # Add this line if you want to display the plot in addition to saving it


def dice_coefficient(prediction, ground_truth):
    """!Calculate the Dice Similarity Coefficient (DSC) between the model's prediction and the ground truth mask.
    @param prediction: The model's prediction.
    @param ground_truth: The ground truth mask.
    @return: The DSC score.
    """
    smooth = 1  # To avoid division by zero
    prediction = torch.sigmoid(
        prediction
    )  # Applying sigmoid to convert logits to probabilities
    prediction = prediction > 0.5  # Thresholding to obtain binary prediction map
    intersection = (prediction * ground_truth).sum()
    dice = (2.0 * intersection + smooth) / (
        prediction.sum() + ground_truth.sum() + smooth
    )
    return dice


def predict_random_test_case(
    model, device, dataset_path="image_datasets.pth", threshold=0.5
):
    """! Load a random test case from the dataset, display the selected image and its ground truth mask,
    and make a prediction using the given model.

    Parameters:
    @param model: The trained model used for making predictions.
    @param device: The device (CPU or GPU) to perform computations on.
    @param dataset_path: Path to the dataset file. Defaults to 'image_datasets.pth'.
    @param threshold: Threshold for converting model outputs to binary predictions. Defaults to 0.5.
    """
    # Load the dataset
    image_datasets = torch.load(dataset_path, map_location=device)
    test_cases = image_datasets["val"]

    # Randomly select a test case
    random_case = random.choice(test_cases)
    mask_array, image_array = random_case[1], random_case[0]
    image_tensor = torch.Tensor(image_array)
    mask_tensor = torch.Tensor(mask_array)

    curr_image, curr_mask = image_tensor[0, :, :], mask_tensor[0, :, :]

    # Display the selected slice
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(curr_image, cmap="gray")
    plt.title("Randomly selected input image from test set", wrap=True)
    plt.subplot(1, 3, 2)
    plt.imshow(curr_mask, cmap="jet")
    plt.title("Corresponding ground truth mask", wrap=True)

    # Prepare the image for prediction
    curr_image = curr_image.unsqueeze(0).unsqueeze(0).to(device)

    # Make prediction
    curr_pred = model(curr_image)
    curr_pred = torch.sigmoid(curr_pred)
    curr_pred = (curr_pred > threshold).float().cpu()[0, 0, :, :]

    # Display the model prediction
    plt.subplot(1, 3, 3)
    plt.imshow(curr_pred, cmap="jet")
    plt.title("Model prediction", wrap=True)
    plt.tight_layout()
    plt.savefig("plots/random_example_pred.png")
