from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision
import torch
from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH
from gan_compare.data_utils.utils import load_inbreast_mask

import pydicom as dicom
from typing import Tuple
from pathlib import Path
import json
import os.path
import glob
import cv2
import numpy as np


class InbreastDataset(Dataset):
    """Inbreast dataset."""

    def __init__(
        self, 
        metadata_path: str,
        crop: bool = True, 
        min_size: int = 160, 
        margin: int = 100, 
        final_shape: Tuple[int, int] = (400, 400),
    ):
        assert Path(metadata_path).is_file(), "Metadata not found"
        with open(metadata_path, "r") as metadata_file:
            self.metadata = json.load(metadata_file)
        self.crop = crop
        self.min_size = min_size
        self.margin = margin
        self.final_shape = final_shape
    
    def __len__(self):
        # metadata contains a list of lesion objects (incl. patient id, bounding box, etc)
        return len(self.metadata)
    
    def _convert_to_uint8(self, image: np.ndarray) -> np.ndarray:
        # normalize value range between 0 and 255 and convert to 8-bit unsigned integer
        img_n = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
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
        if xml_filepath != "":
            with open(xml_filepath, "rb") as patient_xml:
                mask = load_inbreast_mask(patient_xml, ds.pixel_array.shape)
        else:
            mask = np.zeros(ds.pixel_array.shape)
        image = self._convert_to_uint8(ds.pixel_array)
        mask = mask.astype('uint8')
        x, y, w, h = self._get_crops_around_mask(metapoint)
        image, mask = image[y:y+h, x:x+w], mask[y:y+h, x:x+w]
        # scale
        image = cv2.resize(image, self.final_shape, interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask, self.final_shape, interpolation = cv2.INTER_AREA)
        if to_save:
            return image

        # sample = {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}
        sample = [torchvision.transforms.functional.to_tensor(image[..., np.newaxis])]

        return sample
