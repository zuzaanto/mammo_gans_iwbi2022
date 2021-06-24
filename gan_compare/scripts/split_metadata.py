import os
from pathlib import Path
import json
import argparse
import numpy as np
from typing import Tuple


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


def split_array_into_folds(metadata_array: np.ndarray, proportion: float) -> Tuple[np.ndarray]:
    fold2_index = np.random.uniform(size=len(metadata_array)) > proportion
    fold2 = metadata_array[fold2_index]
    fold1 = metadata_array[~fold2_index]
    return fold1, fold2


def save_metadata_to_file(metadata_array: np.ndarray, out_path: Path) -> None:
    if len(metadata_array) > 0:
        if not out_path.parent.exists():
            os.makedirs(out_path.parent)
        with open(str(out_path.resolve()), "w") as out_file:
            json.dump(metadata_array.tolist(), out_file, indent=4)
            

if __name__ == "__main__":
    args = parse_args()
    metadata = None
    with open(args.metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    print(f"Number of metapoints: {len(metadata)}")
    # val_metadata = random.sample(metadata, int(args.split_metadata ** len(metadata))
    metadata_array = np.array(metadata)
    train_metadata, remaining_metadata = split_array_into_folds(metadata_array, args.train_proportion)
    val_proportion_scaled = args.val_proportion / (1. - args.train_proportion)
    val_metadata, test_metadata = split_array_into_folds(remaining_metadata, val_proportion_scaled)
    print(f"Split metadata into {len(train_metadata)}, {len(val_metadata)} and {len(test_metadata)} samples.")
    print("Saving..")
    out_dirpath = Path(args.output_dir)
    save_metadata_to_file(train_metadata, out_dirpath.joinpath("train_metadata.json"))
    save_metadata_to_file(val_metadata, out_dirpath.joinpath("val_metadata.json"))
    save_metadata_to_file(test_metadata, out_dirpath.joinpath("test_metadata.json"))
    print(f"Saved train, test and val metadata to separate files in {out_dirpath.resolve()}")
