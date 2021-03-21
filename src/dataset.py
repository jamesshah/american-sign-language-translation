import numpy as np
import torch
from PIL import Image
import albumentations


class ImageDataset:
    def __init__(self, image_paths, targets, augmentations=None, channel_first=True):

        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.channel_first = channel_first

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        targets = self.targets[idx]
        image = Image.open(self.image_paths[idx])
        image = np.array(image)

        augmented = self.augmentations(image=image)
        image = augmented["image"]

        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
