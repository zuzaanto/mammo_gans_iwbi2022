import pandas as pd
import json
import argparse
from pathlib import Path
from gan_compare.data_utils.utils import save_metadata_to_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--in_first_metadata", type=str, required=True, help="Path to first metadata file to merge",
    )
    parser.add_argument(
        "--in_second_metadata", type=str, required=True, help="Path to second metadata file to merge",
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Path to output merged metadata to",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    first_metadata = pd.read_json(args.in_first_metadata)
    second_metadata = pd.read_json(args.in_second_metadata)
    merged_metadata = first_metadata.append(second_metadata)
    merged_metadata.healthy = merged_metadata.healthy.fillna(False)
    assert len(first_metadata) + len(second_metadata) == len(merged_metadata)
    save_metadata_to_file(merged_metadata, out_path=Path(args.out_path))
    print(f"Saved {len(first_metadata)} + {len(second_metadata)} = {len(merged_metadata)} merged metadata to {Path(args.out_path).resolve()}")


if __name__ == "__main__":
    main()