from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision

from gan_compare.data_utils.utils import load_inbreast_mask, convert_to_uint8, get_crops_around_mask
from gan_compare.dataset.base_dataset import BaseDataset
from gan_compare.dataset.constants import BIRADS_DICT, DENSITY_DICT


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
            split_birads_fours: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            is_trained_on_calcifications: bool = False,
            is_trained_on_masses: bool = True,
            is_trained_on_other_roi_types: bool = False,
            is_condition_binary: bool = False,
            is_condition_categorical:bool=False,
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
            split_birads_fours=split_birads_fours,
            is_trained_on_calcifications=is_trained_on_calcifications,
            is_trained_on_masses=is_trained_on_masses,
            is_trained_on_other_roi_types=is_trained_on_other_roi_types,
            is_condition_binary=is_condition_binary,
            is_condition_categorical=is_condition_categorical,
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
            condition = self.resolve_and_get_condition(metapoint=metapoint)
            if is_image_returned:
                return sample, image, condition
            else:
                return sample, condition

        if is_image_returned:
            return sample, image
        else:
            return sample

    def resolve_and_get_condition(self, metapoint):
        condition = None
        if self.conditional:
            if self.conditioned_on == "birads":
                if self.is_condition_binary:
                    condition = metapoint["birads"][0]
                    if int(condition) <= 3:
                        return 0
                    return 1
                elif self.split_birads_fours:
                    condition = BIRADS_DICT[metapoint["birads"]]
                else:
                    condition = metapoint["birads"][
                        0
                    ]  # avoid 4c, 4b, 4a and just truncate them to 4
                # We could also have evaluation of is_condition_categorical here if we want continuous birads not
                # either 0 or 1 (0 or 1 is already provided by setting the is_condition_binary to true)
            elif self.conditioned_on == "density":
                if self.is_condition_binary:
                    condition = metapoint["density"][0]
                    if int(float(condition)) <= 2:
                        return 0
                    return 1
                elif self.is_condition_categorical:
                    condition = metapoint["density"][0]  # 1-4
                else:  # return a value between 0 and 1 using the DENSITY_DICT.
                    condition: float = DENSITY_DICT[metapoint["density"][0]]
        return condition
