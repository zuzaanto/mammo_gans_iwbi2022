import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision
from torch.utils.data import Dataset

from gan_compare.data_utils.utils import load_inbreast_mask

BIRADS_DICT = {
    "2": 1,
    "3": 2,
    "4a": 3,
    "4b": 4,
    "4c": 5,
    "5": 6,
    "6": 7,
}


class InbreastDataset(Dataset):
    """Inbreast dataset."""

    def __init__(
            self,
            metadata_path: str,
            crop: bool = True,
            min_size: int = 160,
            margin: int = 100,
            final_shape: Tuple[int, int] = (400, 400),
            conditional_birads: bool = False,
            split_birads_fours: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            is_trained_on_calcifications: bool = False,
            is_trained_on_masses: bool = True,
            is_trained_on_other_roi_types: bool = False,
            transform: any = None,
    ):
        assert Path(metadata_path).is_file(), f"Metadata not found in {metadata_path}"
        self.metadata = []
        with open(metadata_path, "r") as metadata_file:
            metadata_unfiltered = json.load(metadata_file)
        assert is_trained_on_masses or is_trained_on_calcifications or is_trained_on_other_roi_types, \
            f"You specified to train the GAN neither on masses nor calcifications nor other roi types. Please select " \
            f"at least one roi type. "
        if is_trained_on_masses:
            self.metadata.extend(
                [metapoint for metapoint in metadata_unfiltered if metapoint['roi_type'] == 'Mass'])
            print(f'Appended Masses to metadata. Metadata size: {len(self.metadata)}')

        if is_trained_on_calcifications:
            self.metadata.extend(
                [metapoint for metapoint in metadata_unfiltered if metapoint['roi_type'] == 'Calcification'])
            print(f'Appended Calcifications to metadata. Metadata size: {len(self.metadata)}')

        if is_trained_on_other_roi_types:
            self.metadata.extend(
                [metapoint for metapoint in metadata_unfiltered if metapoint['roi_type'] == 'Other'])
            print(f'Appended Other ROI types to metadata. Metadata size: {len(self.metadata)}')

        self.crop = crop
        self.min_size = min_size
        self.margin = margin
        self.final_shape = final_shape
        self.conditional_birads = conditional_birads
        self.split_birads_fours = split_birads_fours
        self.transform = transform

    def __len__(self):
        # metadata contains a list of lesion objects (incl. patient id, bounding box, etc)
        return len(self.metadata)

    def _convert_to_uint8(self, image: np.ndarray) -> np.ndarray:
        # normalize value range between 0 and 255 and convert to 8-bit unsigned integer
        img_n = cv2.normalize(
            src=image,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        return img_n

    def _get_crops_around_mask(self, metapoint: dict) -> Tuple[int, int]:
        x, y, w, h = metapoint["bbox"]
        # pad the bbox
        x_p = max(0, x - self.margin // 2)
        y_p = max(0, y - self.margin // 2)
        w_p = w + self.margin
        h_p = h + self.margin
        # make sure the bbox is bigger than min size
        if w_p < self.min_size:
            x_p = max(0, x - (self.min_size - w_p) // 2)
            w_p = self.min_size
        if h_p < self.min_size:
            y_p = max(0, y - (self.min_size - h_p) // 2)
            h_p = self.min_size
        return (x_p, y_p, w_p, h_p)

    def __getitem__(self, idx: int, to_save: bool = False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        image_path = metapoint["image_path"]
        ds = dicom.dcmread(image_path)
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
        image = self._convert_to_uint8(ds.pixel_array)
        mask = mask.astype("uint8")
        x, y, w, h = self._get_crops_around_mask(metapoint)
        image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]

        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)
        if to_save:
            if self.conditional_birads:
                condition = f"{metapoint['birads'][0]}"
                return image, condition
            else:
                return image

        # sample = {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}

        sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        # print(sample)
        if self.transform:
            sample = self.transform(sample)

        if self.conditional_birads:
            if self.split_birads_fours:
                condition = BIRADS_DICT[metapoint["birads"]]
            else:
                condition = metapoint["birads"][
                    0
                ]  # avoid 4c, 4b, 4a and just truncate them to 4
            return sample, int(condition)

        return sample
