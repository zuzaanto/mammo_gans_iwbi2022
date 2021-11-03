from typing import Tuple

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import torchvision

from gan_compare.data_utils.utils import get_crops_around_mask, retrieve_condition
from gan_compare.dataset.base_dataset import BaseDataset


class BCDRDataset(BaseDataset):
    """BCDR dataset."""

    def __init__(
            self,
            metadata_path: str,
            crop: bool = True,
            min_size: int = 160,
            margin: int = 100,
            final_shape: Tuple[int, int] = (400, 400),
            conditioned_on: str = None,
            conditional: bool = False,
            is_condition_binary: bool = False,
            is_condition_categorical: bool = False,
            split_birads_fours: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            is_trained_on_calcifications: bool = False,
            is_trained_on_masses: bool = True,
            is_trained_on_other_roi_types: bool = False,
            transform: any = None,
    ):
        super().__init__(
            metadata_path=metadata_path,
            crop=crop,
            min_size=min_size,
            margin=margin,
            final_shape=final_shape,
            conditioned_on=conditioned_on,
            conditional=conditional,
            is_condition_binary=is_condition_binary,
            is_condition_categorical=is_condition_categorical,
            split_birads_fours=split_birads_fours,
            is_trained_on_calcifications=is_trained_on_calcifications,
            is_trained_on_masses=is_trained_on_masses,
            is_trained_on_other_roi_types=is_trained_on_other_roi_types,
            transform=transform,
        )
        assert is_trained_on_masses or is_trained_on_calcifications or is_trained_on_other_roi_types, \
            f"You specified to train the GAN neither on masses nor calcifications nor other roi types. Please select " \
            f"at least one roi type. "
        if is_trained_on_masses:
            self.metadata.extend(
                [metapoint for metapoint in self.metadata_unfiltered if "nodule" in metapoint["roi_type"]])
            print(f'Appended Masses to metadata. Metadata size: {len(self.metadata)}')

        if is_trained_on_calcifications:
            # TODO add these keywords to a dedicated constants file
            self.metadata.extend(
                [metapoint for metapoint in self.metadata_unfiltered \
                 if "calcification" in metapoint["roi_type"] \
                 or "microcalcification" in metapoint["roi_type"]
                 ]
            )
            print(f'Appended Calcifications to metadata. Metadata size: {len(self.metadata)}')

        if is_trained_on_other_roi_types:
            self.metadata.extend(
                [metapoint for metapoint in self.metadata_unfiltered \
                 if "axillary_adenopathy" in metapoint["roi_type"] \
                 or "architectural_distortion" in metapoint["roi_type"] \
                 or "stroma_distortion" in metapoint["roi_type"]
                 ]
            )
            print(f'Appended Other ROI types to metadata. Metadata size: {len(self.metadata)}')

    def __getitem__(self, idx: int, to_save: bool = False, is_image_returned: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert metapoint.get("dataset") == "bcdr", "Dataset name mismatch, you're using a wrong metadata file!"
        image_path = metapoint["image_path"]
        # TODO read as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        contour = np.asarray(metapoint["contour"])
        # Create an empty image to store the masked array
        r_mask = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype='bool')
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        r_mask[np.round(contour[1, :]).astype('int'), np.round(contour[0, :]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        mask = ndimage.binary_fill_holes(r_mask[:image.shape[0], :image.shape[1]])
        mask = mask.astype("uint8")
        x, y, w, h = get_crops_around_mask(metapoint, margin=self.margin, min_size=self.min_size)
        image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]
        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)

        sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])

        if self.transform:
            sample = self.transform(sample)
        if self.conditional:
            condition = retrieve_condition(metapoint=metapoint, conditioned_on=self.conditioned_on,
                                           is_condition_binary=self.is_condition_binary,
                                           is_condition_categorical=self.is_condition_categorical,
                                           split_birads_fours=self.split_birads_fours)
            if is_image_returned:
                return sample, image, condition
            else:
                return sample, condition

        if is_image_returned:
            return sample, image
        else:
            return sample
