"""@file unet.py

@brief Defines the U-Net model and training function for the lung segmentation pipeline.

@details The U-Net model and training function are defined in this file.
The U-Net model is a simple implementation of the U-Net architecture for semantic segmentation.
The training function is used to train the U-Net model on the lung segmentation dataset.
The training function uses the Binary Cross-Entropy (BCE) loss and the Dice loss as the loss function,
and the Adam optimizer with learning rate scheduling.

@author Created by C. Grivot on 14/03/2024
"""
import torch
import torch.nn as nn
import time
from torchmetrics.classification import BinaryAccuracy


class SimpleUNet(nn.Module):
    """! Class representing the U-Net model for lung segmentation."""

    def __init__(self, in_channels, out_channels):
        """! Constructor method.
        @param in_channels The number of input channels.
        @param out_channels The number of output channels.
        """
        super().__init__()
        self.conv1 = self.conv_block(in_channels, 16, 3, 1, 1)
        self.maxpool1 = self.maxpool_block(2, 2, 0)
        self.conv2 = self.conv_block(16, 32, 3, 1, 1)
        self.maxpool2 = self.maxpool_block(2, 2, 0)
        self.conv3 = self.conv_block(32, 64, 3, 1, 1)
        self.maxpool3 = self.maxpool_block(2, 2, 0)
        self.middle = self.conv_block(64, 128, 3, 1, 1)
        self.upsample3 = self.transposed_block(128, 64, 3, 2, 1, 1)
        self.upconv3 = self.conv_block(128, 64, 3, 1, 1)
        self.upsample2 = self.transposed_block(64, 32, 3, 2, 1, 1)
        self.upconv2 = self.conv_block(64, 32, 3, 1, 1)
        self.upsample1 = self.transposed_block(32, 16, 3, 2, 1, 1)
        self.upconv1 = self.conv_block(32, 16, 3, 1, 1)
        self.final = self.final_layer(16, 1, 1, 1, 0)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """! Function for convolutional block.
        @param in_channels The number of input channels.
        @param out_channels The number of output channels.
        @param kernel_size The size of the kernel.
        @param stride The stride of the kernel.
        @param padding The padding of the kernel.
        """
        convolution = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return convolution

    def maxpool_block(self, kernel_size, stride, padding):
        """! Function for maxpooling block.
        @param kernel_size The size of the kernel.
        @param stride The stride of the kernel.
        @param padding The padding of the kernel.
        """
        maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout2d(0.5),
        )
        return maxpool

    def transposed_block(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        """! Function for transposed convolutional block.
        @param in_channels The number of input channels.
        @param out_channels The number of output channels.
        @param kernel_size The size of the kernel.
        @param stride The stride of the kernel.
        @param padding The padding of the kernel.
        @param output_padding The output padding of the kernel.
        """
        transposed = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        return transposed

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """! Function to define the final layer.
        @param in_channels The number of input channels.
        @param out_channels The number of output channels.
        @param kernel_size The size of the kernel.
        @param stride The stride of the kernel.
        @param padding The padding of the kernel.
        """
        final = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        return final

    def forward(self, x):
        """! Function to forward pass the input through the U-Net model.
        @param x The input tensor.
        """
        # downsampling part
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # middle part
        middle = self.middle(maxpool3)
        # upsampling part
        upsample3 = self.upsample3(middle)
        upconv3 = self.upconv3(torch.cat([upsample3, conv3], 1))
        upsample2 = self.upsample2(upconv3)
        upconv2 = self.upconv2(torch.cat([upsample2, conv2], 1))
        upsample1 = self.upsample1(upconv2)
        upconv1 = self.upconv1(torch.cat([upsample1, conv1], 1))
        final_layer = self.final(upconv1)
        return final_layer


class CustomLoss(nn.Module):
    """! Custom loss function combining Binary Cross-Entropy (BCE) loss and Dice loss."""

    def __init__(self, weights=[0.5, 0.5]):
        """! Constructor method.
        @param weights The weights for the BCE and Dice loss.
        """
        super(CustomLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weights = weights

    def dice_loss(self, inputs, targets, smooth=1e-6):
        """! Function to calculate the Dice loss.
        @param inputs: The input tensor.
        @param targets: The target tensor.
        @param smooth: The smoothing factor.
        """
        # Apply sigmoid activation to predict probabilities
        inputs = torch.sigmoid(inputs)
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        # Calculate Dice loss
        dice = 1 - (2.0 * intersection + smooth) / (union + smooth)
        return dice

    def forward(self, inputs, targets):
        """! Function to forward pass the input through the loss function.
        @param inputs The input tensor.
        @param targets The target tensor."""
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        # Return weighted sum of BCE and Dice loss
        return self.weights[0] * bce + self.weights[1] * dice


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    device,
    dataset_sizes,
    num_epochs=25,
):
    """! Function to train the U-Net model.
    @param model The U-Net model.
    @param criterion The loss function.
    @param optimizer The optimizer.
    @param scheduler The learning rate scheduler.
    @param dataloaders The data loaders for training and validation.
    @param device The device to use for training.
    @param dataset_sizes The sizes of the training and validation datasets.
    @param num_epochs The number of epochs to train for.
    """
    since = time.time()

    all_acc = []
    losses = []
    # Train model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0

        # Initialize accuracy metric for training
        metric = BinaryAccuracy(threshold=0.5).to(device)

        # Iterate over training data.
        for inputs, masks in dataloaders["train"]:
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            metric.update(outputs, masks)

            # Statistics
            running_loss += loss.item() * inputs.size(0)

        # Compute accuracy for this epoch
        acc = metric.compute()
        metric.reset()  # Reset metric for next epoch

        # Scheduler step
        scheduler.step()

        epoch_loss = running_loss / dataset_sizes["train"]
        epoch_acc = acc.cpu().item()

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        losses.append(epoch_loss)
        all_acc.append(epoch_acc)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Save final model weights
    torch.save(model.state_dict(), "SimpleUNet_new.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "accuracies": all_acc,
    }

    torch.save(checkpoint, "model_checkpoint.pth")

    return model
