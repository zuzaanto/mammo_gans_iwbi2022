from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision

from gan_compare.data_utils.utils import load_inbreast_mask, convert_to_uint8
from gan_compare.dataset.base_dataset import BaseDataset


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
            config = None
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
            config=config
        )
        if self.classify_binary_healthy:
            self.metadata.extend(
                [metapoint for metapoint in self.metadata_unfiltered if metapoint['dataset'] == 'inbreast'])
            print(f'Appended InBreast metadata. Metadata size: {len(self.metadata)}')
        else:
            assert is_trained_on_masses or is_trained_on_calcifications or is_trained_on_other_roi_types, \
                f"You specified to train the GAN neither on masses nor calcifications nor other roi types. Please select " \
                f"at least one roi type. "
            if is_trained_on_masses:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Mass'])
                print(f'Appended Masses to metadata. Metadata size: {len(self.metadata)}')

            if is_trained_on_calcifications:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Calcification'])
                print(f'Appended Calcifications to metadata. Metadata size: {len(self.metadata)}')

            if is_trained_on_other_roi_types:
                self.metadata.extend(
                    [metapoint for metapoint in self.metadata_unfiltered if metapoint['roi_type'] == 'Other'])
                print(f'Appended Other ROI types to metadata. Metadata size: {len(self.metadata)}')


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
            #         print(
            #             f"Error when trying to mask_list[0].get('mask'). mask_list: {mask_list}, metapoint: {metapoint}")
        # else:
            # mask = np.zeros(ds.pixel_array.shape)
            # print(f"xml_filepath Error for metapoint: {metapoint}")
        if metapoint.get("healthy", False):
            y, x, w, h = metapoint["bbox"]
            # x, y, w, h = self.get_crops_around_bbox(metapoint["bbox"], margin=0, min_size=self.min_size, image_shape=image.shape, config=self.config)
            image = image[y: y + h, x: x + w]
            # print(f"image.shape: {image.shape}")
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
        
        label = self.retrieve_condition(metapoint) if self.conditional else self.determine_label(metapoint)
        
        return sample, label, image
