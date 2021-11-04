from abc import abstractmethod
import json
from pathlib import Path
from typing import Tuple, List

from torch.utils.data import Dataset
import random

from gan_compare.dataset.constants import BIRADS_DICT

# TODO add option for shuffling in data from synthetic metadata file

class BaseDataset(Dataset):
    """Abstract dataset class."""

    def __init__(
        self,
        metadata_path: str,
        crop: bool = True,
        min_size: int = 160,
        margin: int = 100,
        final_shape: Tuple[int, int] = (400, 400),
        conditional_birads: bool = False,
        classify_binary_healthy: bool = False,
        split_birads_fours: bool = False,  # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
        is_trained_on_calcifications: bool = False,
        is_trained_on_masses: bool = True,
        is_trained_on_other_roi_types: bool = False,
        is_condition_binary:bool = False,
        transform: any = None,
    ):
        assert Path(metadata_path).is_file(), f"Metadata not found in {metadata_path}"
        self.metadata = []
        with open(metadata_path, "r") as metadata_file:
            self.metadata_unfiltered = json.load(metadata_file)
        self.is_condition_binary = is_condition_binary
        self.crop = crop
        self.min_size = min_size
        self.margin = margin
        self.final_shape = final_shape
        self.classify_binary_healthy = classify_binary_healthy
        self.conditional_birads = conditional_birads
        self.split_birads_fours = split_birads_fours
        self.transform = transform


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def determine_label(self, metapoint):
        label = None
        if self.classify_binary_healthy:
            label = int(metapoint.get("healthy", False)) # label = 1 iff metapoint is healthy
        elif self.conditional_birads:
            if self.is_condition_binary:
                condition = metapoint["birads"][0]
                if int(condition) <= 3: label = 0
                else: label = 1
            elif self.split_birads_fours:
                condition = BIRADS_DICT[metapoint["birads"]]
                label = int(condition)
            else:
                condition = metapoint["birads"][0] # avoid 4c, 4b, 4a and just truncate them to 4
                label = int(condition)
        # else: None
        return label