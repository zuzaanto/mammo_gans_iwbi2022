from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision

from gan_compare.data_utils.utils import convert_to_uint8
from gan_compare.dataset.base_dataset import BaseDataset

import logging

class CBIS_DDSMDataset(BaseDataset):
    """CBIS-DDSM dataset."""

    def __init__(
            self,
            metadata_path: str,
            crop: bool = True,
            min_size: int = 128,
            margin: int = 60,
            conditional_birads: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            transform: any = None,
            config = None
    ):
        super().__init__(
            metadata_path=metadata_path,
            crop=crop,
            min_size=min_size,
            margin=margin,
            conditional_birads=conditional_birads,
            transform=transform,
            config=config
        )
        if self.config.classify_binary_healthy:
            self.metadata.extend(
                [metapoint for metapoint in self.metadata_unfiltered if metapoint['dataset'] == 'cbis-ddsm'])
            logging.info(f'Appended CBIS-DDSM metadata. Metadata size: {len(self.metadata)}')
        else:
            assert self.config.is_trained_on_masses or self.config.is_trained_on_calcifications or self.config.is_trained_on_other_roi_types, \
                f"You specified to train the GAN neither on masses nor calcifications nor other roi types. Please select " \
                f"at least one roi type. "
            if self.config.is_trained_on_masses:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Mass' and metapoint['dataset'] == 'cbis-ddsm'])
                logging.info(f'Appended CBIS-DDSM Masses to metadata. Metadata size: {len(self.metadata)}')

            if self.config.is_trained_on_calcifications:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Calcification' and metapoint['dataset'] == 'cbis-ddsm'])
                logging.info(f'Appended CBIS-DDSM Calcifications to metadata. Metadata size: {len(self.metadata)}')

            if self.config.is_trained_on_other_roi_types:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Other' and metapoint['dataset'] == 'cbis-ddsm'])
                logging.info(f'Appended CBIS-DDSM Other ROI types to metadata. Metadata size: {len(self.metadata)}')


    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert metapoint.get("dataset") == "cbis-ddsm", "Dataset name mismatch, you're using a wrong metadata file!"
        image_path = metapoint["image_path"]
        ds = dicom.dcmread(image_path)
        image = convert_to_uint8(ds.pixel_array)
        if metapoint.get("healthy", False):
            x, y, w, h = self.get_crops_around_bbox(metapoint["bbox"], margin=0, min_size=self.min_size, image_shape=image.shape, config=self.config)
            image = image[y: y + h, x: x + w]
            # logging.info(f"image.shape: {image.shape}")
        else:
            # mask = mask.astype("uint8")
            x, y, w, h = self.get_crops_around_bbox(metapoint['bbox'], margin=self.margin, min_size=self.min_size, image_shape=image.shape, config=self.config)
            # image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]
            image = image[y: y + h, x: x + w]

            
        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        # mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)

        sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])

        if self.transform: sample = self.transform(sample)
        
        label = self.retrieve_condition(metapoint) if self.config.conditional else self.determine_label(metapoint)
        
        return sample, label, image, metapoint['roi_type']
