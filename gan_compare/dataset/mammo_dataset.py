import logging
from typing import Optional

import cv2
import numpy as np
import pydicom as dicom
import torch
import torchvision

from gan_compare.data_utils.utils import convert_to_uint8
from gan_compare.dataset.base_dataset import BaseDataset
from gan_compare.dataset.metapoint import Metapoint
from gan_compare.paths import DATASET_PATH_DICT
from gan_compare.training.base_config import BaseConfig
from gan_compare.training.io import load_json


class MammographyDataset(BaseDataset):
    """Mammography dataset class."""

    def __init__(
        self,
        metadata_path: str,
        config: BaseConfig,
        split_path: Optional[str] = None,
        subset: str = "train",
        crop: bool = True,
        min_size: int = 128,
        margin: int = 60,
        conditional_birads: bool = False,
        # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
        transform: any = None,
        sampling_ratio: float = 1.0,
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
        )

        if self.config.classify_binary_healthy:
            assert split_path is not None, "Missing split path!"
        split_dict = load_json(split_path)
        self.patient_ids = split_dict[subset]
        self.metadata.extend(
            [
                metapoint
                for metapoint in self.metadata_unfiltered
                if metapoint.patient_id in self.patient_ids
            ]
        )
        # filter datasets of interest
        self.metadata = [
            metapoint
            for metapoint in self.metadata
            if metapoint.dataset in self.config.data[subset].dataset_names
        ]
        # filter roi types of interest
        self.metadata = [
            metapoint
            for metapoint in self.metadata
            if any(
                roi_type in self.config.data[subset].roi_types
                for roi_type in metapoint.roi_type
            )
        ]
        logging.info(f"Appended metadata. Metadata size: {len(self.metadata)}")

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metapoint = self.metadata[idx]
        assert isinstance(metapoint, Metapoint)
        image_path = str(
            (DATASET_PATH_DICT[metapoint.dataset] / metapoint.image_path).resolve()
        )
        if image_path.endswith(".dcm"):
            ds = dicom.dcmread(image_path)
            image = convert_to_uint8(ds.pixel_array)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if metapoint.healthy:
            x, y, w, h = self.get_crops_around_bbox(
                metapoint.bbox,
                margin=0,
                min_size=self.min_size,
                image_shape=image.shape,
                config=self.config,
            )
            image = image[y : y + h, x : x + w]
            # logging.info(f"image.shape: {image.shape}")
        else:
            # mask = mask.astype("uint8")
            x, y, w, h = self.get_crops_around_bbox(
                metapoint.bbox,
                margin=self.margin,
                min_size=self.min_size,
                image_shape=image.shape,
                config=self.config,
            )
            # image, mask = image[y: y + h, x: x + w], mask[y: y + h, x: x + w]
            image = image[y : y + h, x : x + w]
        if self.config.model_name == "swin_transformer":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # scale
        image = cv2.resize(image, self.final_shape, interpolation=cv2.INTER_AREA)
        # mask = cv2.resize(mask, self.final_shape, interpolation=cv2.INTER_AREA)
        if self.config.model_name != "swin_transformer":
            sample = torchvision.transforms.functional.to_tensor(image[..., np.newaxis])
        else:
            sample = torchvision.transforms.functional.to_tensor(image)

        if self.transform:
            sample = self.transform(sample)

        label = (
            self.retrieve_condition(metapoint)
            if self.config.conditional
            else self.determine_label(metapoint)
        )
        return sample, label, image, metapoint.roi_type[0], metapoint.patch_id
