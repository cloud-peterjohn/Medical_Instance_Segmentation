import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import os


def compute_mean_std(root_dir, image_folder="train"):
    print(
        f"Calculating mean and standard deviation for dataset {root_dir}/{image_folder}..."
    )

    # Construct the path to the training image folder
    train_dir = Path(root_dir) / image_folder
    image_folders = [f for f in train_dir.iterdir() if f.is_dir()]

    # Initialize accumulators
    pixel_sum = np.zeros(3)  # Sum of RGB channels
    pixel_squared_sum = np.zeros(3)  # Sum of squared RGB channels
    pixel_count = 0

    # Iterate through all images
    for img_folder in tqdm(image_folders):
        img_path = img_folder / "image.tif"
        if not img_path.exists():
            print(f"Warning: Image {img_path} does not exist")
            continue

        # Read the image
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Unable to read image {img_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Convert the image to float32 for higher precision
            image = image.astype(np.float32) / 255.0

            # Accumulate pixel values
            pixel_sum += np.sum(image, axis=(0, 1))
            pixel_squared_sum += np.sum(image**2, axis=(0, 1))
            pixel_count += image.shape[0] * image.shape[1]
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    # Calculate mean
    mean = pixel_sum / pixel_count

    # Calculate standard deviation
    # Using the formula: std = sqrt(E[X^2] - (E[X])^2)
    var = (pixel_squared_sum / pixel_count) - (mean**2)
    std = np.sqrt(var)

    return mean, std


if __name__ == "__main__":
    # Path to the dataset root directory
    root_dir = "hw3-data-release"  # Modify to the actual dataset path

    # Compute the mean and standard deviation of the dataset
    mean, std = compute_mean_std(root_dir)

    print("\nResults:")
    print(f"Mean (RGB): {mean.tolist()}")
    print(f"Standard Deviation (RGB): {std.tolist()}")
    print("\nTo use in code:")
    print(f"Normalize(mean={mean.tolist()}, std={std.tolist()})")

    # Save results to a text file
    with open("dataset_stats.txt", "w") as f:
        f.write(f"Mean (RGB): {mean.tolist()}\n")
        f.write(f"Standard Deviation (RGB): {std.tolist()}\n")
