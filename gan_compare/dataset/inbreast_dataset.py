from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision
from pathlib import Path

from gan_compare.data_utils.utils import convert_to_uint8
from gan_compare.dataset.base_dataset import BaseDataset

import logging

class InbreastDataset(BaseDataset):
    """Inbreast dataset."""

    def __init__(
            self,
            metadata_path: str,
            crop: bool = True,
            min_size: int = 128,
            margin: int = 60,
            conditional_birads: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            transform: any = None,
            config=None
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
                [metapoint for metapoint in self.metadata_unfiltered if metapoint['dataset'] == 'inbreast'])
            logging.info(f'Appended InBreast metadata. Metadata size: {len(self.metadata)}')
        else:
            assert self.config.is_trained_on_masses or self.config.is_trained_on_calcifications or self.config.is_trained_on_other_roi_types, \
                f"You specified to train the GAN neither on masses nor calcifications nor other roi types. Please select " \
                f"at least one roi type. "
            if self.config.is_trained_on_masses:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Mass'])
                logging.info(f'Appended Masses to metadata. Metadata size: {len(self.metadata)}')

            if self.config.is_trained_on_calcifications:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Calcification'])
                logging.info(f'Appended Calcifications to metadata. Metadata size: {len(self.metadata)}')

            if self.config.is_trained_on_other_roi_types:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Other'])
                logging.info(f'Appended Other ROI types to metadata. Metadata size: {len(self.metadata)}')

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert metapoint.get("dataset") == "inbreast", "Dataset name mismatch, you're using a wrong metadata file!"
        image_path = metapoint["image_path"]
        ds = dicom.dcmread(image_path)
        if ds is None:
            logging.warning(
                f"image in path {image_path} was not read in properly. Is file there (?): {Path(image_path).is_file()}. "
                f"Fallback: Using next file at index {idx + 1} instead. Please check your metadata file.")
            return self.__getitem__(idx + 1)
        image = convert_to_uint8(ds.pixel_array)

        # xml_filepath = metapoint["xml_path"]
        # expected_roi_type = metapoint["roi_type"]
        # if xml_filepath != "":
        # with open(xml_filepath, "rb") as patient_xml:
        #     mask_list = load_inbreast_mask(patient_xml, ds.pixel_array.shape, expected_roi_type=expected_roi_type)
        #     try:
        #         mask = mask_list[0].get('mask')
        #     except:
        #         print(
        #             f"Error when trying to mask_list[0].get('mask'). mask_list: {mask_list}, metapoint: {metapoint}")
        # else:
        # mask = np.zeros(ds.pixel_array.shape)
        # print(f"xml_filepath Error for metapoint: {metapoint}")

        if "healthy" not in metapoint or metapoint.get("healthy", False):
            x, y, w, h = self.get_crops_around_bbox(metapoint["bbox"], margin=0, min_size=self.min_size,
                                                    image_shape=image.shape, config=self.config)
            image = image[y: y + h, x: x + w]
            # logging.info(f"image.shape: {image.shape}")
        else:
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

        if self.transform: sample = self.transform(sample)
        
        label = self.retrieve_condition(metapoint) if self.config.conditional else self.determine_label(metapoint)
        
        return sample, label, image, metapoint['roi_type']
