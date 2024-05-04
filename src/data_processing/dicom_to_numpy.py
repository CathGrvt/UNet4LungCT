"""!@file dicom_to_numpy.py

@biref Convert DICOM files to numpy arrays

@details This script contains functions to convert DICOM files to numpy arrays.
The script includes functions to load the DICOM files, convert the files to Hounsfield units,
anonymize the metadata, and save the files as .npy files.
The script also includes a main function that is called when the script is run,
which loads the DICOM files, converts the files to numpy arrays, and saves the files as .npy files.
The main function also includes a try-except block to catch any exceptions
that occur during the conversion and log them before exiting.

@author Created by C. Grivot on 14/03/2024
"""
import pydicom
from pydicom import dcmread
import os
from os import listdir
from os.path import join
import numpy as np


def load_scan(path):
    """! Load and sort DICOM images from a directory.
    @param path The directory containing the DICOM images.
    """
    slices = [
        pydicom.dcmread(join(path, s)) for s in listdir(path) if s.endswith(".dcm")
    ]

    # # Attempt to sort by InstanceNumber first (Doesn't match with segmentation files)
    # try:
    #     slices.sort(key=lambda x: int(x.InstanceNumber))
    # except AttributeError:
    #     # Fallback to sorting by the z-coordinate of ImagePositionPatient
    #     slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    return slices


def get_pixels_hu(slices):
    """! Convert DICOM images to Hounsfield units and convert them to a numpy array.
    @param slices The DICOM images to convert.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # # Set outside-of-scan pixels to 0
    # image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def anonymize_metadata(path):
    """! Anonymize the metadata of DICOM images in a directory.
    @param path The directory containing the DICOM images.
    """
    for file_name in listdir(path):
        dicom_file = join(path, file_name)
        metadata = dcmread(dicom_file)
        # Modify the tags that contain patient information
        case_id = os.path.basename(path)  # folder name is the case ID
        metadata.PatientID = case_id
        metadata.PatientName = case_id
        metadata.PatientBirthDate = ""
        try:
            del metadata.PatientBirthTime
        except Exception:
            pass
        metadata.save_as(dicom_file)


def convert_to_np(dataset_path):
    """! Convert DICOM files to numpy arrays (full pipeline) and save them as .npy files.
    @param dataset_path The directory containing the DICOM images.
    """
    for case_id in range(0, 12):
        path = f"Case_{case_id:03d}"
        subdir_path = join(dataset_path, path)
        anonymize_metadata(subdir_path)  # Anonymize metadata
        slices = load_scan(subdir_path)  # Load and sort DICOM images
        # Making sure that it is sorted the same way as the segmentations
        slices = sorted(slices, key=lambda s: s.SliceLocation)
        image_array = get_pixels_hu(slices)  # Convert to HU and numpy array
        np.save(
            f"Dataset/processed/Case_{case_id:03d}.npy", image_array
        )  # Save as .npy file
        print(f"Case {case_id} : ", image_array.shape)
