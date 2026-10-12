from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.unet_model import TinyUNet


# Training settings
DATA_DIR = PROJECT_ROOT / "data" / "processed"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 256
WEIGHTS_PATH = PROJECT_ROOT / "data" / "weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")


class BUSIDataset(Dataset):
    """Load preprocessed BUSI images and their matching binary masks."""

    def __init__(self, image_dir: Path, mask_dir: Path) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(image_dir.glob("*.png"))
        self.image_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

        if not self.images:
            raise FileNotFoundError(
                f"No PNG images found in {image_dir}. Run backend/preprocess.py first."
            )

        missing_masks = [
            image_path.name
            for image_path in self.images
            if not self._mask_path(image_path).is_file()
        ]
        if missing_masks:
            examples = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Missing masks for {len(missing_masks)} images in {mask_dir}: {examples}"
            )

    def _mask_path(self, image_path: Path) -> Path:
        # preprocess.py prefixes both files with the category and adds "_mask"
        # to the mask filename.
        return self.mask_dir / f"{image_path.stem}_mask.png"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self._mask_path(image_path)

        with Image.open(image_path) as source_image:
            image = self.image_transform(source_image.convert("RGB"))
        with Image.open(mask_path) as source_mask:
            mask = self.mask_transform(source_mask.convert("L"))

        # Treat all annotated mask pixels as foreground.
        mask = (mask > 0).float()
        return image, mask


def main() -> None:
    dataset = BUSIDataset(
        image_dir=DATA_DIR / "images",
        mask_dir=DATA_DIR / "masks",
    )
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TinyUNet().to(DEVICE)
    # TinyUNet returns sigmoid probabilities, so use BCELoss (not logits loss).
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training with {len(dataset)} image-mask pairs.")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)
            loss = criterion(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_loss = epoch_loss / len(train_loader)
        print(
            f"--- Epoch {epoch + 1} complete. "
            f"Average loss: {average_loss:.4f} ---"
        )

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Training complete! Weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
