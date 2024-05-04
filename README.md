# Medical Imaging - cg845

This repository contains a UNet-based segmentation model for medical imaging, specifically focusing on lung segmentation in CT scans. It also includes tools for handling DICOM data, which is a standard format for storing medical imaging information.

## Table of contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Running the Full Pipeline](#running-the-full-pipeline)
4. [Features](#features)
5. [Frameworks](#frameworks)
6. [Credits](#credits)

## Requirements

This project uses Docker to maintain a consistent environment across various systems. Please note the following requirements and recommendations for a smooth setup:

- **Memory Considerations:** The process may get terminated in the Docker container if there is insufficient memory available on your personal laptop. Ensure that you are running the project on a machine with adequate memory resources.

- **Tested Platforms:** The process has been tested and functions normally on AWS compute resources. It is also expected to work well on High-Performance Computing (HPC) environments.

- **Docker Usage:** Make sure to have Docker installed and configured properly on your system. Follow the official Docker documentation for installation and setup instructions. If Docker is unavailable, using Conda is recommended for managing dependencies. Ensure that all packages listed in `environment.yml` are installed if neither Docker nor Conda is an option.

## Setup

### Using Docker

Build the Docker image with:

```bash
docker build -t cg845_unet .
```
The setup image will also add the necessary pre-commit checks to your git repository, ensuring the commits work correctly. Ensure you are in a Git repository directory.

Run the Docker container:

```bash
docker run --name <container_name> --rm -ti cg845_unet
```
Once running the container, ensure that you have the right conda environment activated: "conda activate unet".

To use Git within the container, mount your SSH directory:

```bash
docker run --name <container_name> --rm -v <local_ssh_dir>:/root/.ssh -ti cg845_unet
```

### Using Conda

Create and activate the Conda environment:

```bash
conda env create --name <env_name> -f environment.yml
conda activate <env_name>
```
Make sure to clone the Dataset repository:

```bash
git clone https://github.com/loressa/DataScience_MPhill_practicals.git && \
    mv DataScience_MPhill_practicals/Dataset ./Dataset && \
    mkdir -p Dataset/processed/ && \
    rm -rf DataScience_MPhill_practicals
```
## Running the Full Pipeline

The pipeline includes data preprocessing, model training, evaluation, and visualisation steps. To start the pipeline, run:

```bash
python run_pipeline.py
```

Ensure that configuration files are correctly set for specifying paths and parameters.
Ensure that you have the right conda environment activated: "conda activate unet".
To optimize the performance and reduce the execution time of the pipeline, it is recommended to run it on a GPU instead of a CPU.
## Features

### The UNet-based Segmentation

- Implements a UNet architecture for accurate lung segmentation in CT scans.
- Customisable model parameters through configuration files, for now this file is only used to run the whole pipeline.

### Visualisation

- Tools for visualising segmentation results alongside ground truth for comparison.
- Supports plotting of loss and accuracy metrics over training epochs.

### Documentation

- Auto-generated documentation using Doxygen. Run `doxygen` in the `/docs` folder to generate.
- This README serves as the front page for the documentation.



## Frameworks

This project uses **Python** with key dependencies including:

- **Computation and parsing:** numpy, pydicom, torch, torchmetrics
- **Plotting:** matplotlib
- **Maintainability/documentation:** doxygen, pytest, pre-commit


## Credits

- `.pre-commit-config.yaml` adapted from Research Computing lectures.
- Dataset from the Lung CT Segmentation Challenge 2017 (LCTSC), via The Cancer Imaging Archive (TCIA), distributed under the Creative Commons Attribution 3.0 Unported License.


## Appendix: Use of Auto-generation Tools

### ChatGPT

Throughout the development of this project, ChatGPT by OpenAI was utilized for various purposes including code generation, prototyping, debugging assistance, and conceptual explanations. Below, I outline the instances of ChatGPT's use, detailing the nature of the prompts provided, the context in which the output was employed, and any modifications made to the generated content.

#### Code Generation and Prototyping
- **Prompts Submitted**: Queries were made to generate Python code snippets for loading and processing DICOM images, understanding a 3D CNN model in PyTorch, and if it was needed to modify the 2D CNN model, and writing custom dataset classes for PyTorch DataLoader.
- **Usage**: Generated code was used as a foundation for the project's data preprocessing pipeline and model implementation. This included loading DICOM images, converting them to tensors, and defining the neural network architecture.
- **Modifications**: The code provided by ChatGPT was adapted to fit the specific requirements of the dataset and project objectives, on its own it didn't function with the provided dataset and context. This involved adjusting data loading mechanisms from my part to handle the dataset's unique structure, and optimizing performance.

#### Debugging Assistance
- **Prompts Submitted**: Assistance was requested for debugging issues related to Docker file, Custom Dataset, DataLoader, including errors with variable image sizes and tensor stacking.
- **Usage**: Suggestions from ChatGPT were employed to resolve runtime errors and improve the data loading process.
- **Modifications**: Debugging advice was integrated with existing code, with adjustments made to accommodate the specific data formats and processing goals.

### Drafting and Proofreading
- **Prompts Submitted**: Requests were made for drafting sections of the README file and technical documentation, as well as for proofreading and suggesting alternative wordings.
- **Usage**: The output was utilised to enhance the clarity and completeness of project documentation.
- **Modifications**: Generated text was revised to better align with the coursework's scope, terminology, and presentation style.

### Co-Pilot
- **Usage**: GitHub Copilot was used for code suggestion, completion, and documentation.

### Declaration
This appendix serves to acknowledge the use of ChatGPT and GitHub Copilot in the development of this coursework, in line with the guidance provided in the handbook. The contributions of these auto-generation tools have been explicitly cited, detailing their application and the modifications made to their outputs.

In instances where the output of these tools was utilized without modification, it was done so with the understanding that the generated content met the project's requirements accurately. The use of these tools was instrumental in facilitating the project's development, offering insights, and expediting the coding process.

Should there be any sections of this project where the use of auto-generation tools is not cited, it is to be understood that those sections were developed independently of such assistance.
