"""@file test_eval.py

@brief Test that the evaluation functions return the expected results.

@details This file contains the unit tests for the evaluation functions of the lung segmentation pipeline. The tests include the evaluation of the model, the computation of the IoU and Dice scores, and the visualization of the metrics.

@author Created by C. Grivot on 14/03/2024
"""
from src.utilities.metrics import evaluate_model, compute_iou, dice_coefficient
import torch


def test_evaluate_model():
    """! Test the evaluate_model function to ensure it returns correct dice and IoU scores.
    """

    # Mocks
    class MockModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, inputs):
            # Mock prediction to be all ones, simulating a perfect match with the true_masks for testing purposes
            return torch.ones_like(inputs)

    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10  # Arbitrary number of samples

        def __getitem__(self, idx):
            # Return inputs and true_masks as tensors full of ones
            return torch.ones((1, 64, 64)), torch.ones((1, 64, 64))

    test_loader = torch.utils.data.DataLoader(MockDataset(), batch_size=1)
    model = MockModel()

    # Execution
    dice_scores, iou_scores, accuracy_scores = evaluate_model(model, test_loader)

    # Assertions
    assert all(
        0.9 < score <= 1.0 for score in dice_scores
    ), "Dice scores should be close to 1 for perfect overlap."
    assert all(
        0.9 < score <= 1.0 for score in iou_scores
    ), "IoU scores should be close to 1 for perfect overlap."
    assert all(
        score == 1.0 for score in accuracy_scores
    ), "Accuracy scores should be 1 for perfect overlap."


def test_compute_iou():
    """! Test compute_iou function with predefined prediction and ground truth to ensure correct IoU calculation.
    """
    prediction_logits = torch.tensor(
        [[[[0.9]]]]
    )  # Logits high enough to ensure a prediction of 1
    ground_truth = torch.tensor([[[[1]]]])  # Ground truth is 1

    # Execution
    iou_score = compute_iou(prediction_logits, ground_truth)

    # Assertion
    assert iou_score == 1.0, "IoU score should be 1.0 for perfect match."


def test_dice_coefficient():
    """! Test dice_coefficient function with predefined prediction and ground truth to ensure correct Dice score calculation.
    """
    prediction = torch.tensor(
        [[[[0.9]]]]
    )  # Probability high enough to ensure a prediction of 1
    ground_truth = torch.tensor([[[[1]]]])  # Ground truth is 1

    # Execution
    dice_score = dice_coefficient(prediction, ground_truth)

    # Assertion
    assert dice_score > 0.9, "Dice score should be close to 1 for perfect match."
