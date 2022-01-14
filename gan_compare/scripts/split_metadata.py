import os
from pathlib import Path
import json
import argparse
import numpy as np
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split
import pandas as pd
from gan_compare.constants import DENSITIES
from gan_compare.data_utils.utils import save_metadata_to_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path", required=True, type=str, help="Path to json file with metadata."
    )    
    parser.add_argument(
        "--train_proportion", default=0.7, type=float, help="Proportion of train subset."
    )    
    parser.add_argument(
        "--val_proportion", default=0.15, type=float, help="Proportion of val subset."
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Directory to save 3 new metadata files."
    )
    args = parser.parse_args()
    return args


def split_df_into_folds(metadata_df: pd.DataFrame, proportion: float) -> Tuple[pd.DataFrame]:
    fold2_index = np.random.uniform(size=len(metadata_df)) > proportion
    fold2 = metadata_df[fold2_index]
    fold1 = metadata_df[~fold2_index]
    return fold1, fold2


def split_array_into_folds(patients_list: np.ndarray, proportion: float) -> Tuple[np.ndarray]:
    fold2_index = np.random.uniform(size=len(patients_list)) > proportion
    fold2 = patients_list[fold2_index]
    fold1 = patients_list[~fold2_index]
    return fold1, fold2


if __name__ == "__main__":
    args = parse_args()
    metadata = None
    metadata_df = pd.read_json(args.metadata_path)
    print(f"Number of metapoints: {len(metadata_df)}")
    # TODO add option to append existing metadata
    train_metadata = pd.DataFrame([])
    val_metadata = pd.DataFrame([])
    test_metadata = pd.DataFrame([])
    used_patients = []
    for density in DENSITIES:
        metadata_per_density = metadata_df[metadata_df.density == density]
        patients = metadata_per_density.patient_id.unique()
        train_patients_per_density, remaining_patients = split_array_into_folds(patients, args.train_proportion)
        val_proportion_scaled = args.val_proportion / (1. - args.train_proportion)
        val_patients_per_density, test_patients_per_density = split_array_into_folds(remaining_patients, val_proportion_scaled)

        train_patients = train_patients_per_density.tolist()
        val_patients = val_patients_per_density.tolist()
        test_patients = test_patients_per_density.tolist()

        for train_patient in train_patients:
            if train_patient not in used_patients:
                train_metadata = train_metadata.append(metadata_per_density[metadata_per_density.patient_id == train_patient])
        used_patients.extend(train_patients)
        for val_patient in val_patients:
            if val_patient not in used_patients:
                val_metadata = val_metadata.append(metadata_per_density[metadata_per_density.patient_id == val_patient])
        used_patients.extend(val_patients)
        for test_patient in test_patients:
            if test_patient not in used_patients:
                test_metadata = test_metadata.append(metadata_per_density[metadata_per_density.patient_id == test_patient])
        used_patients.extend(test_patients)

    # Some metapoints may not contain density label - we don't want them in any of the splits
    assert (len(train_metadata) + len(val_metadata) + len(test_metadata)) <= len(metadata_df)

    print(f"Split metadata into {len(train_metadata)}, {len(val_metadata)} and {len(test_metadata)} samples.")
    print("Saving..")
    out_dirpath = Path(args.output_dir)
    save_metadata_to_file(train_metadata, out_dirpath.joinpath("train_metadata.json"))
    save_metadata_to_file(val_metadata, out_dirpath.joinpath("val_metadata.json"))
    save_metadata_to_file(test_metadata, out_dirpath.joinpath("test_metadata.json"))
    print(f"Saved train, test and val metadata to separate files in {out_dirpath.resolve()}")
