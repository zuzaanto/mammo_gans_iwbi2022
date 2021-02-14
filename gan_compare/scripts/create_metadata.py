from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH
from gan_compare.data_utils.utils import load_inbreast_mask, get_file_list

from typing import Tuple
from pathlib import Path
import os.path
import glob
import cv2
import numpy as np
import json
import argparse


def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_path", required=True, help="Path to json file to store metadata in."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    metadata = []
    image_path = os.path.join(INBREAST_IMAGE_PATH,
                              get_file_list()[idx])
    patient_id = Path(image_path).stem.split("_")[0]
    xml_filepath = Path(INBREAST_XML_PATH) / f"{patient_id}.xml"
    if xml_filepath.is_file():
        with open(xml_filepath, "rb") as patient_xml:
            mask = load_inbreast_mask(patient_xml, ds.pixel_array.shape)
    else:
        mask = np.zeros(ds.pixel_array.shape)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    mask2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lesion_areas = []
    for c in contours:
        lesion_areas.append(cv2.boundingRect(c))