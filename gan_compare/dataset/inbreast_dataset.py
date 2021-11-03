from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision

from gan_compare.data_utils.utils import load_inbreast_mask, convert_to_uint8, get_crops_around_mask, retrieve_condition
from gan_compare.dataset.base_dataset import BaseDataset


class InbreastDataset(BaseDataset):
    """Inbreast dataset."""

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

    def __getitem__(self, idx: int, to_save: bool = False, is_image_returned: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert metapoint.get("dataset") == "inbreast", "Dataset name mismatch, you're using a wrong metadata file!"
        image_path = metapoint["image_path"]
        ds = dicom.dcmread(image_path)
        image = convert_to_uint8(ds.pixel_array)
        xml_filepath = metapoint["xml_path"]
        expected_roi_type = metapoint["roi_type"]
        if xml_filepath != "":
            with open(xml_filepath, "rb") as patient_xml:
                mask_list = load_inbreast_mask(patient_xml, ds.pixel_array.shape, expected_roi_type=expected_roi_type)
                try:
                    mask = mask_list[0].get('mask')
                except:
                    print(
                        f"Error when trying to mask_list[0].get('mask'). mask_list: {mask_list}, metapoint: {metapoint}")
        else:
            mask = np.zeros(ds.pixel_array.shape)
            print(f"xml_filepath Error for metapoint: {metapoint}")

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
