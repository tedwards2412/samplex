import numpy as np
import fnmatch
import os
from datetime import datetime
import mlx.core as mx


def get_bestfit(chains):
    flattened_chains = chains.reshape(-1, chains.shape[-1])
    idx = mx.argmin(flattened_chains[:, 0])
    return flattened_chains[idx]


def generate_filename(folder_name):
    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Get the current date in the desired format
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Initialize chain number
    chain_number = 1

    # Pattern to match filenames
    pattern = f"{current_date}__*.npy"

    # List all files in the folder
    existing_files = os.listdir(folder_name)

    # Filter files that match the pattern and find the highest chain number
    for file in existing_files:
        if fnmatch.fnmatch(file, pattern):
            # Extract the chain number from the filename
            existing_chain_number = int(file.split("__")[1].split(".")[0])
            # Update the chain number if a higher number is found
            chain_number = max(chain_number, existing_chain_number + 1)

    # Generate the new filename
    filename = f"{current_date}__{chain_number}.npy"
    return os.path.join(folder_name, filename)
