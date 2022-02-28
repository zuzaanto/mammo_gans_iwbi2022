import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from gan_compare.constants import DATASET_LIST
from gan_compare.data_utils.utils import save_split_to_file
from gan_compare.dataset.constants import DENSITY_DICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path",
        required=True,
        type=str,
        help="Path to json file with metadata.",
    )
    parser.add_argument(
        "--train_proportion",
        default=0.7,
        type=float,
        help="Proportion of train subset.",
    )
    parser.add_argument(
        "--val_proportion", default=0.15, type=float, help="Proportion of val subset."
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Directory to save a split file with patient ids.",
    )
    args = parser.parse_args()
    return args


def split_df_into_folds(
    metadata_df: pd.DataFrame, proportion: float
) -> Tuple[pd.DataFrame]:
    fold2_index = np.random.uniform(size=len(metadata_df)) > proportion
    fold2 = metadata_df[fold2_index]
    fold1 = metadata_df[~fold2_index]
    return fold1, fold2


def split_array_into_folds(
    patients_list: np.ndarray, proportion: float
) -> Tuple[np.ndarray]:
    fold2_index = np.random.uniform(size=len(patients_list)) > proportion
    fold2 = patients_list[fold2_index]
    fold1 = patients_list[~fold2_index]
    return fold1, fold2


if __name__ == "__main__":
    args = parse_args()
    out_path = (
        Path(args.metadata_path).parent / "train_test_val_split.json"
        if args.output_path is None
        else Path(args.output_path)
    )
    metadata = None
    all_metadata_df = pd.read_json(args.metadata_path)
    print(f"Number of metapoints: {len(all_metadata_df)}")
    # TODO add option to append existing metadata
    train_patients_list = []
    val_patients_list = []
    test_patients_list = []
    used_patients = []
    for dataset in DATASET_LIST:  # Split is done separately for each dataset
        dataset_metadata_df = all_metadata_df[all_metadata_df.dataset == dataset]
        metadata_df_masses = dataset_metadata_df[
            dataset_metadata_df.roi_type.apply(lambda x: "mass" in x)
        ]
        metadata_df_healthy = dataset_metadata_df[
            dataset_metadata_df.roi_type.apply(lambda x: "healthy" in x)
        ]
        metadata_df_other_lesions = dataset_metadata_df[
            dataset_metadata_df.roi_type.apply(
                lambda x: "mass" not in x and "healthy" not in x
            )
        ]
        for metadata_df in [
            metadata_df_masses,
            metadata_df_healthy,
            metadata_df_other_lesions,
        ]:  # split healthy, masses and other separately to enforce balance
            if len(metadata_df) == 0:
                print(f"Skipping {dataset} metadata as its empty")
                continue
            for density in DENSITY_DICT.keys():
                metadata_per_density = metadata_df[metadata_df.density == density]
                patients = metadata_per_density.patient_id.unique()
                train_patients_per_density, remaining_patients = split_array_into_folds(
                    patients, args.train_proportion
                )
                val_proportion_scaled = args.val_proportion / (
                    1.0 - args.train_proportion
                )
                (
                    val_patients_per_density,
                    test_patients_per_density,
                ) = split_array_into_folds(remaining_patients, val_proportion_scaled)

                train_patients = train_patients_per_density.tolist()
                val_patients = val_patients_per_density.tolist()
                test_patients = test_patients_per_density.tolist()
                train_patients = [
                    train_patient
                    for train_patient in train_patients
                    if train_patient not in used_patients
                ]
                used_patients.extend(train_patients)
                train_patients_list.extend(train_patients)
                val_patients = [
                    val_patient
                    for val_patient in val_patients
                    if val_patient not in used_patients
                ]
                used_patients.extend(val_patients)
                val_patients_list.extend(val_patients)
                test_patients = [
                    test_patient
                    for test_patient in test_patients
                    if test_patient not in used_patients
                ]
                used_patients.extend(test_patients)
                test_patients_list.extend(test_patients)

    # Some metapoints may not contain density label - we don't want them in any of the splits
    assert (
        len(train_patients_list) + len(val_patients_list) + len(test_patients_list)
    ) == len(used_patients)
    # TODO calculate and print statistics of subsets in terms of lesion types, dataset types, densities
    print(
        f"Split patients into {len(train_patients_list)}, {len(val_patients_list)} and {len(test_patients_list)} samples."
    )
    print("Saving..")
    save_split_to_file(
        train_patient_ids=train_patients_list,
        val_patient_ids=val_patients_list,
        test_patient_ids=test_patients_list,
        out_path=out_path,
    )

    print(f"Saved a split file with patient ids to {out_path.resolve()}")
