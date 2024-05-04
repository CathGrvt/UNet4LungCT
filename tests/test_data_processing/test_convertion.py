"""!@file test_convertion.py

@brief Unit tests for conversion

@details This file contains the unit tests for the conversion of DICOM files to numpy arrays.

@author Created by C. Grivot on 14/03/2024
"""
import unittest
import numpy as np
import os
import pydicom
from src.data_processing.dicom_to_numpy import (
    load_scan,
    get_pixels_hu,
    anonymize_metadata,
    convert_to_np,
)


class TestDicomToNumpy(unittest.TestCase):
    """! Class representing the unit tests for the conversion of DICOM files to numpy arrays"""

    def setUp(self):
        """! Set up the test case by initializing the test directory and file paths."""
        self.test_dir = "tests/samples/Case_000/"
        self.test_file = "tests/samples/Case_000/1-001.dcm"
        self.dataset_path = "Dataset/Images"

    def test_load_scan(self):
        """! Test the loading of DICOM files to ensure they are correctly loaded as pydicom datasets."""
        slices = load_scan(self.test_dir)
        self.assertTrue(all(isinstance(s, pydicom.dataset.FileDataset) for s in slices))

    def test_get_pixels_hu(self):
        """! Test the conversion of DICOM files to numpy arrays to ensure they are correctly converted
        to Hounsfield units."""
        slices = load_scan(self.test_dir)
        image = get_pixels_hu(slices)
        self.assertIsInstance(image, np.ndarray)

    def test_anonymize_metadata(self):
        """! Test the anonymization of DICOM files to ensure the patient metadata is correctly anonymized."""
        anonymize_metadata(self.test_dir)
        for file_name in os.listdir(self.test_dir):
            dicom_file = os.path.join(self.test_dir, file_name)
            metadata = pydicom.dcmread(dicom_file)
            self.assertEqual(metadata.PatientID, os.path.basename(self.test_dir))
            self.assertEqual(metadata.PatientName, os.path.basename(self.test_dir))
            self.assertEqual(metadata.PatientBirthDate, "")

    def test_convert_to_np(self):
        """! Test the full conversion pipeline of DICOM files to numpy ,
        to ensure they are correctly saved as .npy files."""
        convert_to_np(self.dataset_path)
        for case_id in range(0, 12):
            np_file = os.path.join("Dataset/processed/", f"Case_{case_id:03d}.npy")
            self.assertTrue(os.path.exists(np_file))


if __name__ == "__main__":
    unittest.main()
