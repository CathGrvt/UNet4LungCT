FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /cg845

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    git \
    jq  # jq is used for editing JSON files in the command line

# Clone the specific dataset directory from GitHub
# Use 'git clone' with '--depth 1' to clone only the latest commit for efficiency
# As git doesn't support cloning specific subdirectories, clone the entire repository
# and remove unnecessary files afterwards if needed
RUN git clone https://github.com/loressa/DataScience_MPhill_practicals.git && \
    mv DataScience_MPhill_practicals/Dataset ./Dataset && \
    rm -rf DataScience_MPhill_practicals

# Copy the rest of your application's source code
COPY . .

# Create the Conda environment
RUN conda env create --name unet -f environment.yml

# Initialize Conda for bash shell
SHELL ["/bin/bash", "--login", "-c"]

# Update the configuration file with the correct dataset paths
# and install pre-commit within the same RUN command to ensure
# the environment is activated.
RUN conda activate unet && \
    jq '.dataset_path = "Dataset/Images" | .mask_dir = "Dataset/Segmentations"' config.json > temp.json && \
    mv temp.json config.json && \
    mkdir -p Dataset/processed/ && \
    pip install pre-commit && \
    pre-commit install

# Ensure commands below run in the created environment by default
ENV PATH /opt/conda/envs/unet/bin:$PATH
