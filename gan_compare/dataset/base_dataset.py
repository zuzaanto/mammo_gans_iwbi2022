import json
import logging
import random
from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset

from gan_compare.dataset.constants import DENSITY_DICT, BIRADS_DICT, BCDR_BIRADS_DICT


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
            added_noise_term: float = 0.0,
            split_birads_fours: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            is_trained_on_calcifications: bool = False,
            is_trained_on_masses: bool = True,
            is_trained_on_other_roi_types: bool = False,
            is_condition_binary: bool = False,
            is_condition_categorical: bool = False,
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
        self.split_birads_fours = split_birads_fours
        self.transform = transform
        self.added_noise_term = added_noise_term

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def retrieve_condition(self, metapoint):
        condition = None
        if self.conditioned_on == "birads":
            try:
                if self.is_condition_binary:
                    condition = metapoint["birads"][0]
                    if int(condition) <= 3:
                        return 0
                    return 1
                elif self.split_birads_fours:
                    condition = int(BIRADS_DICT[metapoint["birads"]])
                else:
                    # avoid 4c, 4b, 4a and just truncate them to 4
                    condition = int(metapoint["birads"][0])
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
                if condition == "N" or int(float(condition)) <= 2:
                    return 0
                return 1
            elif self.is_condition_categorical:
                condition = int(float(metapoint["density"][0]))  # 1-4
            else:  # return a value between 0 and 1 using the DENSITY_DICT.
                # number out of [-1,1] multiplied by noise term parameter. Round for 2 digits
                noise = round(random.uniform(-1, 1) * self.added_noise_term, 2)
                # get the density from the dict and add noise to capture potential variations.
                condition: float = DENSITY_DICT[metapoint["density"][0]]
                # normalising condition between 0 and 1.
                condition = max(min(condition + noise, 1.), 0.)
        return condition
