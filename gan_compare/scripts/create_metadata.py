import argparse
import json
from typing import List

import numpy as np
import pydicom as dicom
from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd

from gan_compare.data_utils.utils import load_inbreast_mask, get_file_list, read_csv, generate_inbreast_metapoints, generate_bcdr_metapoints
from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH, INBREAST_CSV_PATH, BCDR_ROOT_PATH
from gan_compare.dataset.constants import BCDR_SUBDIRECTORIES, BCDR_VIEW_DICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=["inbreast", "bcdr"], 
        nargs="+", 
        help="Name of the dataset of interest.",
    )
    parser.add_argument(
        "--bcdr_subfolder", type=str, nargs="+", default=["d01", "d02"], help="Symbols of BCDR subdirectories to use."
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to json file to store metadata in."
    )
    args = parser.parse_args()
    return args


def create_inbreast_metadata() -> List[dict]:
    metadata = []
    inbreast_df = read_csv(INBREAST_CSV_PATH)
    for filename in tqdm(get_file_list()):
        image_path = INBREAST_IMAGE_PATH / filename
        image_id, patient_id = image_path.stem.split("_")[:2]
        image_id = int(image_id)
        csv_metadata = inbreast_df.loc[inbreast_df["File Name"] == image_id].iloc[0]
        ds = dicom.dcmread(image_path)
        xml_filepath = INBREAST_XML_PATH / f"{image_id}.xml"
        if xml_filepath.is_file():
            with open(xml_filepath, "rb") as patient_xml:
                # returns list of dictionaries, e.g. [{mask:mask1, roi_type:type1}, {mask:mask2, roi_type:type2}]
                mask_list = load_inbreast_mask(patient_xml, ds.pixel_array.shape)
        else:
            mask_list = [{'mask': np.zeros(ds.pixel_array.shape), 'roi_type': ""}]
            print(f'No xml file found. Please review why. Path: {xml_filepath}')
            xml_filepath = ""

        start_index: int = 0
        for mask_dict in mask_list:
            lesion_metapoints, idx = generate_inbreast_metapoints(
                mask=mask_dict.get('mask'), image_id=image_id,
                patient_id=patient_id, csv_metadata=csv_metadata,
                image_path=image_path, xml_filepath=xml_filepath,
                roi_type=mask_dict.get('roi_type'),
                start_index=start_index,
            )
            start_index = idx
            # Add the metapoint objects of each contour to our metadata list
            metadata.extend(lesion_metapoints)
    return metadata

def create_bcdr_metadata(subdirectories_list: List[str]) -> List[dict]:
    assert all(subdir in BCDR_SUBDIRECTORIES.keys() for subdir in subdirectories_list), "Unknown subdirectory symbol"
    metadata = []
    for subdirectory in subdirectories_list:
        csv_outlines_path = BCDR_ROOT_PATH / BCDR_SUBDIRECTORIES[subdirectory] / f"bcdr_{subdirectory}_outlines.csv"
        outlines_df = pd.read_csv(csv_outlines_path)
        img_dir_path = BCDR_ROOT_PATH / BCDR_SUBDIRECTORIES[subdirectory]
        for idx, row in outlines_df.iterrows():
            lesion_metapoint = generate_bcdr_metapoints(image_dir_path=img_dir_path, row_df=row)
            metadata.append(lesion_metapoint)
    return metadata


if __name__ == "__main__":
    args = parse_args()
    metadata = []
    if "inbreast" in args.dataset:
        metadata.extend(create_inbreast_metadata())
    if "bcdr" in args.dataset:
        metadata.extend(create_bcdr_metadata(args.bcdr_subfolder))
    # Output metadata as json file to specified location on disk
    outpath = Path(args.output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath.resolve(), "w") as outfile:
        json.dump(metadata, outfile, indent=4)
    print(f"Saved {len(metadata)} metapoints to {outpath.resolve()}")
