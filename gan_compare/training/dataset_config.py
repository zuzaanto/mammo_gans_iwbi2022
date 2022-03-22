from dataclasses import dataclass, field
from typing import List

from gan_compare.constants import DATASET_LIST
from gan_compare.dataset.constants import ROI_TYPES


@dataclass
class DatasetConfig:
    dataset_names: List[str] = field(default_factory=lambda: DATASET_LIST)
    roi_types: List[str] = field(default_factory=lambda: ROI_TYPES)
    # Add more constraints and filters analogically

    def __post_init__(self):
        assert all(
            dataset_name in DATASET_LIST for dataset_name in self.dataset_names
        ), f"Unrecognized datasets: {self.dataset_names}"
        assert all(
            roi_type in ROI_TYPES for roi_type in self.roi_types
        ), f"Unrecognized roi types: {self.roi_types}"
