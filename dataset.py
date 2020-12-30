import cv2
import torch
import numpy as np
from PIL import Image, ImageFile


class ClassificationDataset:
    def __init__(
            self,
            image_paths,
            targets,
            resize,  # HxW
            augmentations=None,
            backend="pil",
            channel_first=True
    ):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item_idx):
        targets = self.targets[item_idx]

        if self.backend == "pil":
            image = Image.open(self.image_paths[item_idx])
            if self.resize is not None:
                image = image.resize(
                    (self.resize[0], self.resize[1]), resample=Image.BILINEAR
                )
            image = np.array(image)

        elif self.backend == "cv2":
            image = cv2.imread(self.image_paths[item_idx])
            if self.resize is not None:
                image = cv2.resize(
                    image,
                    (self.resize[0], self.resize[1]),
                    interpolation=cv2.INTER_CUBIC
                )
            # image = np.array(image)
        else:
            raise Exception("Backend Not Found")
        # Apply augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        if self.channel_first:
            image = np.reshape(image, (-1, image.shape[0], image.shape[0])).astype(np.float32)

        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets)
        }
