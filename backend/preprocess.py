import os
from pathlib import Path

import cv2
from tqdm import tqdm


# Paths are based on this file, so the script works from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Dataset_BUSI_with_GT"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
IMG_SIZE = 256

# Create folders for the resized images and masks.
os.makedirs(PROCESSED_DATA_PATH / "images", exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH / "masks", exist_ok=True)

processed_count = 0
missing_mask_count = 0

# Process the tumor classes that have segmentation masks.
for category in ["benign", "malignant"]:
    folder_path = RAW_DATA_PATH / category
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    # Ignore masks, including extra mask files such as *_mask_1.png.
    image_files = [
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".png" and "_mask" not in path.stem
    ]

    for image_path in tqdm(image_files, desc=f"Processing {category}"):
        mask_path = image_path.with_name(f"{image_path.stem}_mask.png")

        if not mask_path.exists():
            missing_mask_count += 1
            continue

        # Load the image and grayscale mask.
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not read image: {image_path}")
        if mask is None:
            raise OSError(f"Could not read mask: {mask_path}")

        # Resize masks with nearest-neighbor interpolation to preserve label values.
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Save both files with the category in their names.
        output_image_path = PROCESSED_DATA_PATH / "images" / f"{category}_{image_path.name}"
        output_mask_path = PROCESSED_DATA_PATH / "masks" / f"{category}_{mask_path.name}"
        if not cv2.imwrite(str(output_image_path), image):
            raise OSError(f"Could not save image: {output_image_path}")
        if not cv2.imwrite(str(output_mask_path), mask):
            raise OSError(f"Could not save mask: {output_mask_path}")

        processed_count += 1

print(f"Preprocessing complete! Processed {processed_count} image-mask pairs.")
if missing_mask_count:
    print(f"Skipped {missing_mask_count} images without matching masks.")
