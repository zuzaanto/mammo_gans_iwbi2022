from os import stat
from typing import Optional, Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import random
import torchvision

from gan_compare.data_utils.utils import load_inbreast_mask, convert_to_uint8, get_crops_around_mask
from gan_compare.dataset.base_dataset import BaseDataset
from gan_compare.dataset.constants import BIRADS_DICT


class SyntheticDataset(BaseDataset):
    """Synthetic GAN-generated images dataset."""

    def __init__(
        self,
        metadata_path: str,
        crop: bool = True,
        min_size: int = 160,
        margin: int = 100,
        final_shape: Tuple[int, int] = (400, 400),
        classify_binary_healthy: bool = False,
        conditional_birads: bool = False,
        split_birads_fours: bool = False,  # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
        is_trained_on_calcifications: bool = False,
        is_trained_on_masses: bool = True,
        is_trained_on_other_roi_types: bool = False,
        is_condition_binary:bool = False,
        transform: any = None,
        shuffle_proportion: Optional[int] = None,
        current_length: Optional[int] = None,
    ):
        super().__init__(
            metadata_path=metadata_path,
            crop=crop,
            min_size=min_size,
            margin=margin,
            final_shape=final_shape,
            classify_binary_healthy=classify_binary_healthy,
            conditional_birads=conditional_birads,
            split_birads_fours=split_birads_fours,
            is_trained_on_calcifications=is_trained_on_calcifications,
            is_trained_on_masses=is_trained_on_masses,
            is_trained_on_other_roi_types=is_trained_on_other_roi_types,
            is_condition_binary=is_condition_binary,
            transform=transform,
        )
        # TODO adjust along with synthetic metadata creation
        self.metadata = self.metadata_unfiltered
        assert len(self.metadata) > 0, "Empty synthetic metadata"
        if shuffle_proportion is not None:
            assert current_length is not None, "Cannot calculate expected dataset length without info of current dataset length"
            missing_metadata_count = len(self.metadata) - self._calculate_expected_length(
                current_length=current_length,
                shuffle_proportion=shuffle_proportion
            )
            if missing_metadata_count > 0:
                for idx in range(missing_metadata_count):
                    self.metadata.append(random.choice(self.metadata))
            elif missing_metadata_count < 0:
                self.metadata = random.sample(self.metadata, len(self.metadata + missing_metadata_count))
            print(current_length)
            print(missing_metadata_count)
            print(len(self.metadata))

    @staticmethod
    def _calculate_expected_length(current_length: int, shuffle_proportion: int) -> int:
        return int(shuffle_proportion / (1 - shuffle_proportion) * current_length)

    def __getitem__(self, idx: int, to_save: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        # assert metapoint.get("dataset") == "synthetic", "Dataset name mismatch, you're using a wrong metadata file!"
        image_path = metapoint["image_path"]
        assert ".png" in image_path
        # Synthetic images don't need cropping
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(image.shape)
        mask = mask.astype("uint8")

        if to_save:
            # TODO decide whether we shouldn't be returning a warning here instead
            if self.conditional_birads:
                condition = f"{metapoint['birads'][0]}"
                return image, condition
            else:
                return image

        sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        
        if self.transform: sample = self.transform(sample)

        label = self.determine_label(metapoint)

        return sample, label
