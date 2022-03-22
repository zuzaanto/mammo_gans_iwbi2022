import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision

from gan_compare.dataset.base_dataset import BaseDataset
from gan_compare.training.base_config import BaseConfig


class SyntheticDataset(BaseDataset):
    """Synthetic GAN-generated images dataset."""

    def __init__(
        self,
        config: BaseConfig = None,
        transform: any = None,
    ):
        self.paths = [
            os.path.join(config.synthetic_data_dir, filename)
            for filename in os.listdir(config.synthetic_data_dir)
        ]
        self.config = config
        self.model_name = self.config.model_name
        self.transform = transform

    @staticmethod
    def _calculate_expected_length(current_length: int, shuffle_proportion: int) -> int:
        return int(shuffle_proportion / (1 - shuffle_proportion) * current_length)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int, to_save: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.paths[idx]
        assert ".png" in image_path
        # Synthetic images don't need cropping
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.config.model_name == "swin_transformer":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # mask = np.zeros(image.shape)
        # mask = mask.astype("uint8")

        # if to_save:
        #     # TODO decide whether we shouldn't be returning a warning here instead
        #     if self.conditional_birads:
        #         condition = f"{metapoint['birads'][0]}"
        #         return image, condition
        #     else:
        #         return image
        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        if self.config.model_name != "swin_transformer":
            sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        else:
            sample = torchvision.transforms.functional.to_tensor(image)

        if self.transform:
            sample = self.transform(sample)

        # TODO make more intelligent labels once we have more complex GANs
        label = 0

        return sample, label, image, ""
