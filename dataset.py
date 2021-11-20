import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset


class ImageClassifierDataset(Dataset):
    """Image Classifier dataset"""

    def __init__(
        self,
        img_paths: list,
        lbls: list = None,
        root_dir: str = None,
        transform=None,
        transform_type: str = "",
    ):
        """
        Args:
            img_paths : list containing path of images
            lbls : list containing labels corresponding to images
            root_dir : Parent path for reading images
            transform (callabe, Optional): Transforms to be applied
            transform_type: choices = ['pt', 'albu'] Strategies to apply augmentation
        """
        self.img_filepath = img_paths
        self.lbls = lbls
        self.root_dir = root_dir
        self.transform = transform
        self.transform_type = transform_type

    def __len__(self):
        return len(self.img_filepath)

    def __getitem__(self, idx):
        if self.root_dir is not None:
            img_filename = os.path.join(self.root_dir, self.img_filepath[idx])
        else:
            img_filename = self.img_filepath[idx]
        # apply torchvision transforms
        if self.transform_type == "pt":
            img = Image.open(img_filename).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        # apply albumentation transforms
        elif self.transform_type == "albu":
            img = cv2.imread(img_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                img = self.transform(image=img)["image"]

        if self.lbls is not None:
            return img, self.lbls[idx]
        else:
            return img
