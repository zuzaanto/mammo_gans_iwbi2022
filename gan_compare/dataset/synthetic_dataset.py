from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
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
        conditional_birads: bool = False,
        split_birads_fours: bool = False,  # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
        is_trained_on_calcifications: bool = False,
        is_trained_on_masses: bool = True,
        is_trained_on_other_roi_types: bool = False,
        is_condition_binary:bool = False,
        transform: any = None,
    ):
        super.__init__(
            metadata_path=metadata_path,
            crop=crop,
            min_size=min_size,
            margin=margin,
            final_shape=final_shape,
            conditional_birads=conditional_birads,
            split_birads_fours=split_birads_fours,
            is_trained_on_calcifications=is_trained_on_calcifications,
            is_trained_on_masses=is_trained_on_masses,
            is_trained_on_other_roi_types=is_trained_on_other_roi_types,
            is_condition_binary=is_condition_binary,
            transform=transform,
        )


    def __getitem__(self, idx: int, to_save: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert metapoint.get("dataset") == "synthetic", "Dataset name mismatch, you're using a wrong metadata file!"
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

        # TODO move the following duplicated code to the base class
        if self.transform:
            sample = self.transform(sample)
        if self.conditional_birads:
            if self.is_condition_binary:
                condition = metapoint["birads"][0]
                if int(condition) <= 3:
                    return sample, 0
                return sample, 1
            elif self.split_birads_fours:
                condition = BIRADS_DICT[metapoint["birads"]]
            else:
                condition = metapoint["birads"][
                    0
                ]  # avoid 4c, 4b, 4a and just truncate them to 4
            return sample, int(condition)

        return sample
