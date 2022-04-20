import json
import logging
import random
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from dacite import from_dict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from gan_compare.dataset.constants import (
    BCDR_BIRADS_DICT,
    BIRADS_DICT,
    DENSITY_DICT,
    RADIOMICS_NORMALIZATION_PARAMS,
)
from gan_compare.dataset.metapoint import Metapoint
from gan_compare.training.base_config import BaseConfig


class BaseDataset(Dataset):
    """Abstract dataset class."""

    def __init__(
        self,
        config: BaseConfig,
        metadata_path: str,
        crop: bool = True,
        min_size: int = 128,
        margin: int = 60,
        conditional_birads: bool = False,
        # Setting this to True will result in BiRADS annotation with 4a, 4b, 4c split to separate classes
        transform: any = None,
        sampling_ratio: float = 1.0,
        normalize_output: bool = True,  # applies only to reggresion scenario
        subset: str = "train",
    ):
        assert (
            metadata_path is not None and Path(metadata_path).is_file()
        ), f"Metadata not found in {metadata_path}"
        self.metadata = []
        with open(metadata_path, "r") as metadata_file:
            self.metadata_unfiltered = [
                from_dict(Metapoint, metapoint)
                for metapoint in json.load(metadata_file)
            ]
        logging.info(
            f"Number of {subset} metadata before sampling: {len(self.metadata_unfiltered)}"
        )
        random.seed(config.seed)
        self.metadata_unfiltered = random.sample(
            self.metadata_unfiltered,
            int(sampling_ratio * len(self.metadata_unfiltered)),
        )
        logging.info(
            f"Number of {subset} metadata after sampling: {len(self.metadata_unfiltered)}"
        )

        self.crop = crop
        self.min_size = min_size
        self.margin = margin
        self.conditional_birads = conditional_birads
        self.config = config
        # self.model_name = config.model_name # rather calling self.config.model_name explicitely
        self.final_shape = (self.config.image_size, self.config.image_size)
        self.transform = transform
        self.normalize_output = normalize_output

        if config.is_regression and self.normalize_output:
            self.normalize_output_data(
                self.metadata_unfiltered, self.config.training_target
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def len_of_classes(self):
        """Calculate the number of samples per class. Works only in the binary case.
        Returns:
            (int, int): (num samples of non-healthy, num samples of healthy)
        """
        assert (
            self.config.binary_classification
        ), "Function len_of_classes() works only in the binary classification case."
        cnt = 0
        for d in self.metadata:
            if type(d) is str:
                continue  # then d is a synthetic sample (therefore non-healthy) TODO: might not work the same for synthetic benign/malignant patches
            elif getattr(d, self.config.classes):
                cnt += 1
        return len(self) - cnt, cnt

    def arrange_weights(self, weight_non_healthy, weight_healthy):
        return [
            weight_healthy
            if type(d) is not str and d.is_healthy
            else weight_non_healthy
            for d in self.metadata
        ]

    def retrieve_condition(self, metapoint: Metapoint) -> Union[int, float]:
        condition = -1
        if self.config.conditioned_on == "birads":
            if self.config.is_condition_binary:
                condition = metapoint.birads[0]
                if 0 < int(condition) <= 3:
                    return 0
                elif int(condition) == 0:
                    return metapoint.biopsy_proven_status == "malignant"
                return 1
            elif self.config.split_birads_fours:
                condition = int(BIRADS_DICT[metapoint.birads])
            else:
                # avoid 4c, 4b, 4a and just truncate them to 4
                condition = int(metapoint.birads[0])
            if condition == 0:
                condition = int(BCDR_BIRADS_DICT[metapoint.biopsy_proven_status])
                if self.config.split_birads_fours:
                    condition = int(BIRADS_DICT[str(condition)])

            # We could also have evaluation of is_condition_categorical here if we want continuous birads not
            # to be either 0 or 1 (0 or 1 is already provided by setting the self.is_condition_binary to true)
        elif self.config.conditioned_on == "density":
            if self.config.is_condition_binary:
                condition = metapoint.density
                if condition <= 2:
                    return 0
                return 1
            elif self.is_condition_categorical:
                condition = int(float(metapoint.density))  # 1-4
            else:  # return a value between 0 and 1 using the DENSITY_DICT.
                # number out of [-1,1] multiplied by noise term parameter. Round for 2 digits
                noise = round(random.uniform(-1, 1) * self.config.added_noise_term, 2)
                # get the density from the dict and add noise to capture potential variations.
                condition: float = DENSITY_DICT[metapoint.density]
                # normalising condition between 0 and 1.
                condition = max(min(condition + noise, 1.0), 0.0)
        return condition

    def determine_label(self, metapoint: Metapoint) -> Union[int, float, np.ndarray]:
        target = getattr(metapoint, self.config.training_target)
        if type(target) == int or type(target) == float:
            return target
        if type(target) == dict:
            return np.asarray(list(target.values()))
        if type(target) == bool:
            return int(target)
        if (
            self.config.binary_classification
        ):  # This could be possibly deleted. The bool case above should be enough
            if target == -1:
                raise Exception(
                    f"Target of patch {metapoint.patch_id} is not valid. This happens for example if metapoint.biopsy_proven_status is not set: metapoint.biopsy_proven_status == {metapoint.biopsy_proven_status}"
                )
            else:
                return int(target)  # label = 1 iff metapoint is positive
        elif self.conditional_birads:
            if self.config.is_condition_binary:
                condition = metapoint.birads[0]
                if 0 < int(condition) <= 3:
                    return 0
                elif int(condition) == 0:
                    return metapoint.biopsy_proven_status == "malignant"
                else:
                    return 1
            if int(metapoint.birads[0]) == 0:
                condition = BCDR_BIRADS_DICT[metapoint.biopsy_proven_status]
            else:
                condition = metapoint.birads

            if self.config.split_birads_fours:
                condition = BIRADS_DICT[condition]
                return int(condition)
            else:
                condition = condition[0]  # avoid 4c, 4b, 4a and just truncate them to 4
                return int(condition)
        else:
            return -1  # None does not work

    def get_crops_around_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        margin: int,
        min_size: int,
        image_shape: Tuple[int, int],
        config: BaseConfig,
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox

        x_p, w_p = self.get_measures_for_crop(
            x, w, margin, min_size, image_shape[1], config
        )
        y_p, h_p = self.get_measures_for_crop(
            y, h, margin, min_size, image_shape[0], config, w_p
        )  # second dimension depends on length of first dimension

        return (x_p, y_p, w_p, h_p)

    def get_measures_for_crop(
        self,
        coord,
        length,
        margin,
        min_length,
        image_max,
        config,
        length_of_other_dimension=None,
    ):  # coordinate, length, margin, minimum length, random translation, random zoom
        if length_of_other_dimension is None:
            # Add a margin to the crop:
            l_new = length + 2 * margin  # add one margin left and right each

            # We want to rather zoom out than in
            r_zoom = int(
                np.random.normal(
                    loc=l_new * config.zoom_offset, scale=l_new * config.zoom_spread
                )
            )  # zoom amount depends on the current length of the crop
        else:
            # here we want to set the length and coordinate in relation to the other dimension, to preserve the image ratio
            # We don't add a margin but set l_new to the same value as the other dimension
            coord -= (
                length_of_other_dimension - length
            ) // 2  # required to keep the patch centered
            l_new = length_of_other_dimension
            # r_zoom = int(np.random.normal(loc=0, scale=l_new * config.ratio_spread)) # we still introduce some randomness while preserving image ratio only roughly
            r_zoom = 0  # no randomness anymore

        # Randomly zoom the crop:
        l_new += r_zoom // 2  # random zoom
        coord -= r_zoom // 2  # we want to keep the crop centered here

        # Randomly translate the crop, while it must not be too small or large, or else the lesion won't be within the crop anymore:
        r_transl = min(
            int(-l_new * config.max_translation_offset),
            int(np.random.normal(loc=0, scale=l_new * config.translation_spread)),
        )  # amount depends on the current length of the crop
        if r_transl > int(l_new * config.max_translation_offset):
            r_transl = l_new * config.max_translation_offset

        coord = max(0, coord + r_transl)  # random translation

        # Now make sure that the new length is at least minimum length
        if (
            l_new < min_length
        ):  # => new length is too small, must be at least minimum length
            # explanation: (min_length - l) // 2 > m
            c_new = (
                coord - (min_length - length) // 2
            )  # in this case divide by 2 to keep patch centered
            l_new = min_length
        else:  # => new length is large enough
            c_new = coord - margin

        # Now make sure that the crop is still within the image:
        c_new = max(0, c_new)
        c_new -= max(
            0, (c_new + l_new) - image_max
        )  # move crop back into the image if it goes beyond the image

        return (c_new, l_new)

    def normalize_output_data(self, metadata, attribute):
        data = [
            getattr(metapoint, self.config.training_target) for metapoint in metadata
        ]
        scaler = StandardScaler()

        # load normalization parameters
        if attribute == "radiomics":
            self.norm_feature_params = RADIOMICS_NORMALIZATION_PARAMS
        else:
            # compute normalization parameters (mean, variance)
            self.norm_feature_params = {}
            for feature in data[0].keys():
                per_feature_data = np.array(
                    [features_point[feature] for features_point in data]
                ).reshape(-1, 1)
                scaler.fit(per_feature_data)
                self.norm_feature_params[feature] = (scaler.mean_, scaler.scale_)

        # normalize data using precomputed parameters
        for feature in data[0].keys():
            per_feature_data = np.array(
                [features_point[feature] for features_point in data]
            ).reshape(-1, 1)

            scaler.mean_, scaler.scale_ = self.norm_feature_params[feature]
            norm_per_feature_data = scaler.transform(per_feature_data).squeeze()

            for sample, norm_feature_sample in zip(data, norm_per_feature_data):
                sample[feature] = norm_feature_sample

        for metapoint, norm_sample in zip(metadata, data):
            setattr(metapoint, attribute, norm_sample)

        return metadata
