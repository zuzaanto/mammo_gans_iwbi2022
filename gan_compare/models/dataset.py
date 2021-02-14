from torch.utils.data import Dataset
import torchvision.datasets as dset
import torch
from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH
from gan_compare.data_utils.utils import load_inbreast_mask, get_file_list

import pydicom as dicom
from typing import Tuple
from pathlib import Path
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
        min_size: int = 100, 
        margin: int = 100, 
        final_shape: Tuple[int, int] = (800, 800),
    ):
        assert Path(metadata_path).is_file(), "Metadata not found"
        with open(metadata_path, "r") as metadata_file:
            self.metadata = json.load(metadata_file)
        self.metadata = 
        self.crop = crop
        self.min_size = min_size
        self.margin = margin
        self.final_shape = final_shape
    
    def __len__(self):
        num_files = len(get_file_list())
        # todo fix this to take into account multiple lesions
        return num_files

    def _get_crops_around_mask(self, mask: np.ndarray) -> Tuple[int, int]:
        
        return
    

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(INBREAST_IMAGE_PATH,
                                  get_file_list()[idx])
        ds = dicom.dcmread(image_path)
        patient_id = Path(image_path).stem.split("_")[0]
        xml_filepath = Path(INBREAST_XML_PATH) / f"{patient_id}.xml"
        if xml_filepath.is_file():
            with open(xml_filepath, "rb") as patient_xml:
                mask = load_inbreast_mask(patient_xml, ds.pixel_array.shape)
        else:
            mask = np.zeros(ds.pixel_array.shape)
        image = ds.pixel_array.astype('int64')
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        mask2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lesion_areas = []
        for c in contours:
            lesion_areas.append(cv2.boundingRect(c))
            
        sample = {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample