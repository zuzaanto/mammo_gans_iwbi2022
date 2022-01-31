import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

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
            config=None,
            sampling_ratio: float = 1.0,
            calcifications_only: bool = False,
            masses_only: bool = False,
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
            calcifications_only=calcifications_only,
            masses_only=masses_only
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
        assert metapoint.get("dataset") in ["bcdr",
                                            "bcdr_only_train"], "Dataset name mismatch, you're using a wrong metadata file!"
        image_path = metapoint["image_path"]
        # TODO read as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logging.warning(
                f"image in path {image_path} was not read in properly. Is file there (?): {Path(image_path).is_file()}. "
                f"Fallback: Using next file at index {idx + 1} instead. Please check your metadata file.")
            return self.__getitem__(idx + 1)
        contour = metapoint["contour"]
        if "healthy" not in metapoint or metapoint.get("healthy", False):

            x, y, w, h = self.get_crops_around_bbox(metapoint["bbox"], margin=0, min_size=self.min_size,
                                                    image_shape=image.shape, config=self.config)
            image = image[x: x + h,
                    y: y + w]  # note that the order of axis in healthy bbox is different, TODO change someday

            # logging.info(f"image.shape: {image.shape}")

        elif contour is not None and contour != "NaN":
            contour = np.asarray(contour)
            # Create an empty image to store the masked array
            # logging.info(f"Image path: {image_path}")
            r_mask = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype='bool')
            # Create a contour image by using the contour coordinates rounded to their nearest integer value
            r_mask[np.round(contour[1, :]).astype('int'), np.round(contour[0, :]).astype('int')] = 1
            # Fill in the hole created by the contour boundary
            # mask = ndimage.binary_fill_holes(r_mask[:image.shape[0], :image.shape[1]])
            # mask = mask.astype("uint8")

            x, y, w, h = self.get_crops_around_bbox(metapoint['bbox'], margin=self.margin, min_size=self.min_size,
                                                    image_shape=image.shape, config=self.config)

            # image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]
            image = image[y: y + h, x: x + w]
        # scale
        try:
            image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
            # mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)
        except Exception as e:
            # TODO: Check why some images have a width or height of zero, which causes this exception.
            logging.debug(f"Error in cv2.resize of image (shape: {image.shape}): {e}")
            return None

        sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        if self.transform:
            sample = self.transform(sample)

        label = self.retrieve_condition(metapoint) if self.config.conditional else self.determine_label(metapoint)

        return sample, label, image, metapoint['roi_type']
