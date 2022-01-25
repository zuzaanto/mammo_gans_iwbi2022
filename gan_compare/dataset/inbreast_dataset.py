from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision

from gan_compare.data_utils.utils import load_inbreast_mask, convert_to_uint8
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
            final_shape: Tuple[int, int] = (400, 400),
            classify_binary_healthy: bool = False,
            conditional_birads: bool = False,
            conditioned_on: str = None,
            conditional: bool = False,
            is_condition_binary: bool = False,
            is_condition_categorical: bool = False,
            added_noise_term: float = 0.0,
            split_birads_fours: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            is_trained_on_calcifications: bool = False,
            is_trained_on_masses: bool = True,
            is_trained_on_other_roi_types: bool = False,
            transform: any = None,
            config = None,
            sampling_ratio: float = 1.0,
            calcifications_only: bool = False,
            masses_only: bool = False,
            model_name: str = "cnn"
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
            classify_binary_healthy=classify_binary_healthy,
            conditional_birads=conditional_birads,
            added_noise_term=added_noise_term,
            split_birads_fours=split_birads_fours,
            is_trained_on_calcifications=is_trained_on_calcifications,
            is_trained_on_masses=is_trained_on_masses,
            is_trained_on_other_roi_types=is_trained_on_other_roi_types,
            transform=transform,
            config=config,
            sampling_ratio=sampling_ratio,
            calcifications_only=calcifications_only,
            masses_only=masses_only,
            model_name=model_name,
        )
        if self.classify_binary_healthy:
            self.metadata.extend(
                [metapoint for metapoint in self.metadata_unfiltered if metapoint['dataset'] == 'inbreast'])
            logging.info(f'Appended InBreast metadata. Metadata size: {len(self.metadata)}')
        else:
            assert is_trained_on_masses or is_trained_on_calcifications or is_trained_on_other_roi_types, \
                f"You specified to train the GAN neither on masses nor calcifications nor other roi types. Please select " \
                f"at least one roi type. "
            if is_trained_on_masses:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Mass'])
                logging.info(f'Appended Masses to metadata. Metadata size: {len(self.metadata)}')

            if is_trained_on_calcifications:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Calcification'])
                logging.info(f'Appended Calcifications to metadata. Metadata size: {len(self.metadata)}')

            if is_trained_on_other_roi_types:
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
        image = convert_to_uint8(ds.pixel_array)
        # xml_filepath = metapoint["xml_path"]
        # expected_roi_type = metapoint["roi_type"]
        # if xml_filepath != "":
            # with open(xml_filepath, "rb") as patient_xml:
            #     mask_list = load_inbreast_mask(patient_xml, ds.pixel_array.shape, expected_roi_type=expected_roi_type)
            #     try:
            #         mask = mask_list[0].get('mask')
            #     except:
            #         logging.info(
            #             f"Error when trying to mask_list[0].get('mask'). mask_list: {mask_list}, metapoint: {metapoint}")
        # else:
            # mask = np.zeros(ds.pixel_array.shape)
            # logging.info(f"xml_filepath Error for metapoint: {metapoint}")
        if metapoint.get("healthy", False):
            x, y, w, h = self.get_crops_around_bbox(metapoint["bbox"], margin=0, min_size=self.min_size, image_shape=image.shape, config=self.config)
            image = image[y: y + h, x: x + w]
            # logging.info(f"image.shape: {image.shape}")
        else:
            # mask = mask.astype("uint8")
            x, y, w, h = self.get_crops_around_bbox(metapoint['bbox'], margin=self.margin, min_size=self.min_size, image_shape=image.shape, config=self.config)
            # image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]
            image = image[y: y + h, x: x + w]
        if self.model_name == "swin_transformer":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        # mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)
        if self.model_name != "swin_transformer":
            sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        else:
            sample = torchvision.transforms.functional.to_tensor(image)

        if self.transform: sample = self.transform(sample)
        
        label = self.retrieve_condition(metapoint) if self.conditional else self.determine_label(metapoint)
        
        return sample, label, image, metapoint['roi_type']
