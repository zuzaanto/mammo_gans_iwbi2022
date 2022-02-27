import argparse
import glob
import json
import os.path
import statistics
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pydicom as dicom
from matplotlib import pyplot as plt
from tqdm import tqdm

from gan_compare.data_utils.utils import get_file_list, load_inbreast_mask
from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path", required=True, help="Path to json file with metadata."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    metadata = None
    heights = []
    widths = []
    with open(args.metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    print(f"Number of metapoints: {len(metadata)}")
    for j, metapoint in enumerate(metadata):
        # print(metapoint)
        # print(j)
        widths.append(metapoint["bbox"][2])
        heights.append(metapoint["bbox"][3])
    print(f"Mean (heights): {statistics.mean(heights)}")
    print(f"Mean (widths): {statistics.mean(widths)}")
    print(f"Std dev (heights): {statistics.stdev(heights)}")
    print(f"Std dev (widths): {statistics.stdev(widths)}")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(widths, bins=1000)
    # Show plot
    plt.show()
