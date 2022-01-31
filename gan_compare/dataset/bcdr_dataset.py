from typing import Tuple

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import torchvision
from PIL import Image
import logging

from gan_compare.dataset.base_dataset import BaseDataset


class BCDRDataset(BaseDataset):
    """BCDR dataset."""

    def __init__(
            self,
            metadata_path: str,
            crop: bool = True,
            min_size: int = 128,
            margin: int = 60,
            conditional_birads: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            transform: any = None,
            config = None,
            sampling_ratio: float = 1.0,
    ):
        super().__init__(
            metadata_path=metadata_path,
            crop=crop,
            min_size=min_size,
            margin=margin,
            conditional_birads=conditional_birads,
            transform=transform,
            config=config,
            sampling_ratio=sampling_ratio,
        )
        if self.config.classify_binary_healthy:
            self.metadata.extend(
                [metapoint for metapoint in self.metadata_unfiltered if metapoint['dataset'] == 'bcdr'])
            logging.info(f'Appended BCDR metadata. Metadata size: {len(self.metadata)}')
        else:
            assert self.config.is_trained_on_masses or self.config.is_trained_on_calcifications or self.config.is_trained_on_other_roi_types, \
                f"You specified to train the GAN neither on masses nor calcifications nor other roi types. Please select " \
                f"at least one roi type. "
            if self.config.is_trained_on_masses:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if "nodule" in metapoint["roi_type"]])
                logging.info(f'Appended Masses to metadata. Metadata size: {len(self.metadata)}')

            if self.config.is_trained_on_calcifications:
                # TODO add these keywords to a dedicated constants file
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered \
                    if "calcification" in metapoint["roi_type"] \
                    or "microcalcification" in metapoint["roi_type"]
                    ]
                )
                logging.info(f'Appended Calcifications to metadata. Metadata size: {len(self.metadata)}')

            if self.config.is_trained_on_other_roi_types:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered \
                    if "axillary_adenopathy" in metapoint["roi_type"] \
                    or "architectural_distortion" in metapoint["roi_type"] \
                    or "stroma_distortion" in metapoint["roi_type"]
                    ]
                )
                logging.info(f'Appended Other ROI types to metadata. Metadata size: {len(self.metadata)}')


    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert metapoint.get("dataset") in ["bcdr", "bcdr_only_train"], "Dataset name mismatch, you're using a wrong metadata file!"
        image_path = metapoint["image_path"]
        if self.model_name == "swin_transformer":
            image = cv2.imread(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        contour = metapoint["contour"]
        if metapoint.get("healthy", False):
            x, y, w, h = self.get_crops_around_bbox(metapoint["bbox"], margin=0, min_size=self.min_size, image_shape=image.shape, config=self.config)
            image = image[x: x + h, y: y + w] # note that the order of axis in healthy bbox is different, TODO change someday

            # logging.info(f"image.shape: {image.shape}")

        elif contour is not None:
            contour = np.asarray(contour)
            # Create an empty image to store the masked array
            # logging.info(f"Image path: {image_path}")
            r_mask = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype='bool')
            # Create a contour image by using the contour coordinates rounded to their nearest integer value
            r_mask[np.round(contour[1, :]).astype('int'), np.round(contour[0, :]).astype('int')] = 1
            # Fill in the hole created by the contour boundary
            # mask = ndimage.binary_fill_holes(r_mask[:image.shape[0], :image.shape[1]])
            # mask = mask.astype("uint8")
            x, y, w, h = self.get_crops_around_bbox(metapoint['bbox'], margin=self.margin, min_size=self.min_size, image_shape=image.shape, config=self.config)
            # image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]
            image = image[y: y + h, x: x + w]
        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        # mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)
        if self.model_name != "swin_transformer":
            sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        else:
            sample = torchvision.transforms.functional.to_tensor(image)

        if self.transform: sample = self.transform(sample)

        label = self.retrieve_condition(metapoint) if self.config.conditional else self.determine_label(metapoint)

        roi_type = metapoint["roi_type"] if isinstance(metapoint["roi_type"], str) else metapoint["roi_type"][0]
        
        return sample, label, image, roi_type
