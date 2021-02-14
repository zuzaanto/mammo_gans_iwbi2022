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
from tqdm import tqdm
import pydicom as dicom


def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", required=True, help="Path to json file to store metadata in."
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    metadata = []
    for filename in tqdm(get_file_list()):
        image_path = Path(INBREAST_IMAGE_PATH) / filename
        patient_id = image_path.stem.split("_")[0]
        ds = dicom.dcmread(image_path)
        xml_filepath = Path(INBREAST_XML_PATH) / f"{patient_id}.xml"
        if xml_filepath.is_file():
            with open(xml_filepath, "rb") as patient_xml:
                mask = load_inbreast_mask(patient_xml, ds.pixel_array.shape)
        else:
            mask = np.zeros(ds.pixel_array.shape)
            xml_filepath = ""
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        mask2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lesion_metapoints = []
        for indx, c in enumerate(contours):
            if c.shape[0] < 2:
                continue
            metapoint = {
                "patient_id": patient_id,
                "lesion_id": indx,
                "bbox": cv2.boundingRect(c),
                "image_path": str(image_path.resolve()),        
                "xml_path": str(xml_filepath.resolve()),
                # "contour": c.tolist(),
            }
            lesion_metapoints.append(metapoint)
        metadata.extend(lesion_metapoints)
    with open(args.output_path, "w") as outfile:
        json.dump(metadata, outfile, indent=4)
    print(f"Saved {len(metadata)} metapoints to {args.output_path}")
