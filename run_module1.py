"""@file run_module1.py

@brief Data processing module

@details Run conversion functions to convert DICOM files to numpy arrays.
The script includes a main function that is called when the script is run,
which validates the dataset path, converts the dataset to numpy format,
and logs any exceptions that occur during the conversion before exiting.

@author Created by C. Grivot on 14/03/2024
"""
import logging
import os
import sys
from src.data_processing.dicom_to_numpy import convert_to_np

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_dataset_path(path):
    """!Validate the dataset path.
    @param path: the dataset path to validate
    """
    if not os.path.exists(path):
        logging.error(f"Dataset path does not exist: {path}")
        sys.exit(1)
    if not os.listdir(path):
        logging.error(f"Dataset directory is empty: {path}")
        sys.exit(1)
    logging.info(f"Dataset path validated: {path}")


def main(dataset_path="Dataset/Images"):
    try:
        # Module 1 - Data Processing:
        validate_dataset_path(dataset_path)
        logging.info(f"Starting data conversion on {dataset_path}")
        convert_to_np(dataset_path)
        logging.info("Data conversion completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Getting dataset_path from command-line arguments (if provided)
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset/Images"
    main(dataset_path)
