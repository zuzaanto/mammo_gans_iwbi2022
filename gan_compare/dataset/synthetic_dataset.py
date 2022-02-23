import os
import random
from operator import mod
from os import stat
from typing import Optional, Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision

from gan_compare.data_utils.utils import convert_to_uint8, load_inbreast_mask
from gan_compare.dataset.base_dataset import BaseDataset
from gan_compare.dataset.constants import BIRADS_DICT


class SyntheticDataset(BaseDataset):
    """Synthetic GAN-generated images dataset."""

    def __init__(
        self,
        metadata_path: str = None,
        crop: bool = True,
        min_size: int = 128,
        margin: int = 60,
        conditional_birads: bool = False,
        transform: any = None,
        shuffle_proportion: Optional[int] = None,
        config=None,
        model_name: str = "cnn",
    ):
        super().__init__(
            metadata_path=metadata_path,
            crop=crop,
            min_size=min_size,
            margin=margin,
            conditional_birads=conditional_birads,
            transform=transform,
            config=config,
            model_name=model_name,
        )
        # # TODO adjust along with synthetic metadata creation
        # self.metadata = self.metadata_unfiltered
        # assert len(self.metadata) > 0, "Empty synthetic metadata"
        # if shuffle_proportion is not None:
        #     assert current_length is not None, "Cannot calculate expected dataset length without info of current dataset length"
        #     missing_metadata_count = len(self.metadata) - self._calculate_expected_length(
        #         current_length=current_length,
        #         shuffle_proportion=shuffle_proportion
        #     )
        #     if missing_metadata_count > 0:
        #         for idx in range(missing_metadata_count):
        #             self.metadata.append(random.choice(self.metadata))
        #     elif missing_metadata_count < 0:
        #         self.metadata = random.sample(self.metadata, len(self.metadata + missing_metadata_count))
        #     print(current_length)
        #     print(missing_metadata_count)
        #     print(len(self.metadata))
        self.metadata = [
            os.path.join(config.synthetic_data_dir, filename)
            for filename in os.listdir(config.synthetic_data_dir)
        ]

    @staticmethod
    def _calculate_expected_length(current_length: int, shuffle_proportion: int) -> int:
        return int(shuffle_proportion / (1 - shuffle_proportion) * current_length)

    def __getitem__(self, idx: int, to_save: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # metapoint = self.metadata[idx]
        # assert metapoint.get("dataset") == "synthetic", "Dataset name mismatch, you're using a wrong metadata file!"
        # image_path = metapoint["image_path"]
        image_path = self.metadata[idx]
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

        # label = self.determine_label(metapoint)
        label = 0

        return sample, label, image, ""
