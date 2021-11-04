from abc import abstractmethod
import json
from pathlib import Path
from typing import Tuple, List

from torch.utils.data import Dataset
import logging

from gan_compare.dataset.constants import BCDR_VIEW_DICT, DENSITY_DICT, BIRADS_DICT, BCDR_BIRADS_DICT

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
        conditioned_on: str = None,
        conditional: bool = False,
        conditional_birads: bool = False,
        classify_binary_healthy: bool = False,
        split_birads_fours: bool = False,  # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
        is_trained_on_calcifications: bool = False,
        is_trained_on_masses: bool = True,
        is_trained_on_other_roi_types: bool = False,
        is_condition_binary:bool = False,
        is_condition_categorical:bool = False,
        transform: any = None,
    ):
        assert Path(metadata_path).is_file(), f"Metadata not found in {metadata_path}"
        self.metadata = []
        with open(metadata_path, "r") as metadata_file:
            self.metadata_unfiltered = json.load(metadata_file)
        self.conditioned_on = conditioned_on
        self.is_condition_binary = is_condition_binary
        self.is_condition_categorical = is_condition_categorical
        self.crop = crop
        self.min_size = min_size
        self.margin = margin
        self.final_shape = final_shape
        self.conditional = conditional
        self.classify_binary_healthy = classify_binary_healthy
        self.conditional_birads = conditional_birads
        self.split_birads_fours = split_birads_fours
        self.transform = transform


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def retrieve_condition(self, metapoint):
        condition = None
        if self.conditioned_on == "birads":
            try:
                label = None
                if self.classify_binary_healthy:
                    label = int(metapoint.get("healthy", False)) # label = 1 iff metapoint is healthy
                    condition = label
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
            except Exception as e:
                logging.debug(
                    f"Type Error while trying to extract birads. This could be due to birads field being None in "
                    f"BCDR dataset: {e}. Using biopsy_proven_status field instead as fallback.")
                if self.is_condition_binary:
                    # TODO: Validate if this business logic is desired in experiment,
                    # TODO: e.g. biopsy proven 'Benign' is mapped to BIRADS 3 and Malignant to BIRADS 6
                    condition = BCDR_BIRADS_DICT[metapoint["biobsy_proven_status"]]
                    if int(condition) <= 3:
                        return 0
                    return 1
                elif self.split_birads_fours:
                    condition = int(BIRADS_DICT[str(BCDR_BIRADS_DICT[metapoint["biobsy_proven_status"]])])
                else:
                    condition = int(BCDR_BIRADS_DICT[metapoint["biobsy_proven_status"]])
            # We could also have evaluation of is_condition_categorical here if we want continuous birads not
            # to be either 0 or 1 (0 or 1 is already provided by setting the self.is_condition_binary to true)
        elif self.conditioned_on == "density":
            if self.is_condition_binary:
                condition = metapoint["density"][0]
                # TODO Remove the 'N' comparison after Zuzanna's fix is available
                if not condition == 'N' and int(float(condition)) <= 2:
                    return 0
                return 1
            elif self.is_condition_categorical:
                condition = metapoint["density"][0]  # 1-4
                # TODO Remove the 'N' comparison after Zuzanna's fix is available
                if not condition == 'N':
                    return int(float(condition))
                else:
                    return 3  # TODO This is wrong. Remove after Zuzanna's fix is available
            else:  # return a value between 0 and 1 using the DENSITY_DICT.
                condition: float = DENSITY_DICT[metapoint["density"][0]]
        return condition
