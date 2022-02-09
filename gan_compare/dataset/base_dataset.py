import json
import logging
import random
from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset

from gan_compare.dataset.constants import DENSITY_DICT, BIRADS_DICT, BCDR_BIRADS_DICT

import numpy as np

# TODO add option for shuffling in data from synthetic metadata file

class BaseDataset(Dataset):
    """Abstract dataset class."""

    def __init__(
            self,
            metadata_path: str = None,
            crop: bool = True,
            min_size: int = 128,
            margin: int = 60,
            conditional_birads: bool = False,
            # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
            transform: any = None,
            config = None,
            sampling_ratio: float = 1.0
    ):
        assert Path(metadata_path).is_file(), f"Metadata not found in {metadata_path}"
        self.metadata = []
        with open(metadata_path, "r") as metadata_file:
            self.metadata_unfiltered = json.load(metadata_file)
        logging.info(f"Number of train metadata before sampling: {len(self.metadata_unfiltered)}")
        self.metadata_unfiltered = random.sample(self.metadata_unfiltered, int(sampling_ratio * len(self.metadata_unfiltered)))
        logging.info(f"Number of train metadata after sampling: {len(self.metadata_unfiltered)}")

        self.crop = crop
        self.min_size = min_size
        self.margin = margin
        self.conditional_birads = conditional_birads
        self.config = config
        self.model_name = config.model_name
        self.final_shape = (self.config.image_size, self.config.image_size)
        self.transform = transform


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def retrieve_condition(self, metapoint):
        condition = -1 # None does not work
        if self.config.conditioned_on == "birads":
            try:
                if self.config.is_condition_binary:
                    condition = metapoint["birads"][0]
                    if int(condition) <= 3:
                        return 0
                    return 1
                elif self.config.split_birads_fours:
                    condition = int(BIRADS_DICT[metapoint["birads"]])
                else:
                    # avoid 4c, 4b, 4a and just truncate them to 4
                    condition = int(metapoint["birads"][0])
            except Exception as e:
                logging.debug(
                    f"Type Error while trying to extract birads. This could be due to birads field being None in "
                    f"BCDR dataset: {e}. Using biopsy_proven_status field instead as fallback.")
                if self.config.is_condition_binary:
                    # TODO: Validate if this business logic is desired in experiment,
                    # TODO: e.g. biopsy proven 'Benign' is mapped to BIRADS 3 and Malignant to BIRADS 6
                    condition = BCDR_BIRADS_DICT[metapoint["biopsy_proven_status"]]
                    if int(condition) <= 3:
                        return 0
                    return 1
                elif self.config.split_birads_fours:
                    condition = int(BIRADS_DICT[str(BCDR_BIRADS_DICT[metapoint["biopsy_proven_status"]])])
                else:
                    condition = int(BCDR_BIRADS_DICT[metapoint["biopsy_proven_status"]])
            # We could also have evaluation of is_condition_categorical here if we want continuous birads not
            # to be either 0 or 1 (0 or 1 is already provided by setting the self.is_condition_binary to true)
        elif self.config.conditioned_on == "density":
            if self.config.is_condition_binary:
                condition = metapoint["density"][0]
                if int(float(condition)) <= 2:
                    return 0
                return 1
            elif self.is_condition_categorical:
                condition = int(float(metapoint["density"][0]))  # 1-4
            else:  # return a value between 0 and 1 using the DENSITY_DICT.
                # number out of [-1,1] multiplied by noise term parameter. Round for 2 digits
                noise = round(random.uniform(-1, 1) * self.config.added_noise_term, 2)
                # get the density from the dict and add noise to capture potential variations.
                condition: float = DENSITY_DICT[metapoint["density"][0]] + noise
        return condition

    def determine_label(self, metapoint):
        if self.config.classify_binary_healthy:
            return int(metapoint.get("healthy", False)) # label = 1 iff metapoint is healthy
        elif self.conditional_birads:
            if self.config.is_condition_binary:
                condition = metapoint["birads"][0]
                if int(condition) <= 3: return 0
                else: return 1
            elif self.config.split_birads_fours:
                condition = BIRADS_DICT[metapoint["birads"]]
                return int(condition)
            else:
                condition = metapoint["birads"][0] # avoid 4c, 4b, 4a and just truncate them to 4
                return int(condition)
        else: return -1 # None does not work

    def get_crops_around_bbox(self, bbox: Tuple[int, int, int, int], margin: int, min_size: int, image_shape: Tuple[int, int], config) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox

        x_p, w_p = self.get_measures_for_crop(x, w, margin, min_size, image_shape[1], config)
        y_p, h_p = self.get_measures_for_crop(y, h, margin, min_size, image_shape[0], config, w_p) # second dimension depends on length of first dimension
        
        return (x_p, y_p, w_p, h_p)

    def get_measures_for_crop(self, coord, length, margin, min_length, image_max, config, length_of_other_dimension=None): # coordinate, length, margin, minimum length, random translation, random zoom
        if length_of_other_dimension is None:
            # Add a margin to the crop:
            l_new = length + 2 * margin # add one margin left and right each
        
            # We want to rather zoom out than in
            r_zoom = int(np.random.normal(loc=l_new * config.zoom_offset, scale=l_new * config.zoom_spread)) # zoom amount depends on the current length of the crop
        else:
            # here we want to set the length and coordinate in relation to the other dimension, to preserve the image ratio
            # We don't add a margin but set l_new to the same value as the other dimension
            coord -= (length_of_other_dimension - length) // 2 # required to keep the patch centered
            l_new = length_of_other_dimension
            # r_zoom = int(np.random.normal(loc=0, scale=l_new * config.ratio_spread)) # we still introduce some randomness while preserving image ratio only roughly
            r_zoom = 0 # no randomness anymore

        # Randomly zoom the crop:
        l_new += r_zoom // 2 # random zoom
        coord -= r_zoom // 2 # we want to keep the crop centered here

        # Randomly translate the crop, while it must not be too small or large, or else the lesion won't be within the crop anymore:
        r_transl = min(int(-l_new * config.max_translation_offset), int(np.random.normal(loc=0, scale=l_new * config.translation_spread))) # amount depends on the current length of the crop
        if r_transl > int(l_new * config.max_translation_offset): r_transl = l_new * config.max_translation_offset

        coord = max(0, coord + r_transl) # random translation

        # Now make sure that the new length is at least minimum length
        if l_new < min_length: # => new length is too small, must be at least minimum length
            # explanation: (min_length - l) // 2 > m
            c_new = coord - (min_length - length) // 2 # in this case divide by 2 to keep patch centered
            l_new = min_length
        else: # => new length is large enough
            c_new = coord - margin
        
        # Now make sure that the crop is still within the image:
        c_new = max(0, c_new)
        c_new -= max(0, (c_new + l_new) - image_max) # move crop back into the image if it goes beyond the image

        return (c_new, l_new)