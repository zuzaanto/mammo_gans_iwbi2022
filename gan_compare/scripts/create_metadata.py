import argparse
import json
from typing import List

import numpy as np
import pydicom as dicom
from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd
import cv2

from gan_compare.data_utils.utils import (
    load_inbreast_mask, 
    get_file_list, 
    read_csv, 
    generate_inbreast_metapoints, 
    generate_healthy_inbreast_metapoints,
    generate_bcdr_metapoints, 
    generate_healthy_bcdr_metapoints,
    generate_cbis_ddsm_metapoints
)
from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH, INBREAST_CSV_PATH, BCDR_ROOT_PATH, CBIS_DDSM_ROOT_PATH
from gan_compare.dataset.constants import BCDR_SUBDIRECTORIES, BCDR_VIEW_DICT, BCDR_HEALTHY_SUBDIRECTORIES, CBIS_DDSM_CSV_DICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=["inbreast", "bcdr", "cbis-ddsm"], 
        nargs="+", 
        help="Name of the dataset of interest.",
    )
    parser.add_argument(
        "--healthy", 
        action="store_true", 
        help="Whether to generate healthy metapoints.",
    )
    parser.add_argument(
        "--bcdr_subfolder", type=str, nargs="+", default=["d01", "d02"], help="Symbols of BCDR subdirectories to use."
    )
    parser.add_argument(
        "--cbis_ddsm_csv_set", type=str, nargs="+", default=["train"], help="CBIS-DDSM train/test sets to use."
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to json file to store metadata in."
    )
    parser.add_argument(
        "--per_image_count", type=int, default=5, help="Number of patches to generate from 1 healthy img."
    )
    parser.add_argument(
        "--healthy_size", type=int, default=128, help="Size of patches to generate from 1 healthy img."
    )
    parser.add_argument(
        "--seed", type=int, default=2021, help="Seed the random generator. Can be used for example for generating different healthy patches."
    )
    parser.add_argument(
        "--allowed_calcifications_birads_values", 
        type=str, 
        default=None, # Usage example: ["4a","4b","4c","5","6"]
        nargs="+", 
        help="Name of the birads of interest ONLY FOR CALCIFICATIONS. Other lesions are not affected. If None, all birads values are kept.",
    )
    parser.add_argument(
        "--only_masses", action="store_true", help="Whether to keep only masses. If False, all lesion types are kept.",
    )
    args = parser.parse_args()
    return args


def create_inbreast_metadata(
    healthy: bool = False, 
    per_image_count: int = 5, 
    target_size: int = 128,
    rng = np.random.default_rng(),
    only_masses: bool = False,
    allowed_calcifications_birads_values = []
) -> List[dict]:
    metadata = []
    inbreast_df = read_csv(INBREAST_CSV_PATH)
    for filename in tqdm(get_file_list()):
        image_path = INBREAST_IMAGE_PATH / filename
        image_id, patient_id = image_path.stem.split("_")[:2]
        image_id = int(image_id)
        csv_metadata = inbreast_df.loc[inbreast_df["File Name"] == image_id].iloc[0]
        if healthy:
            start_index: int = 0
            metapoints, idx = generate_healthy_inbreast_metapoints(
                image_id=image_id,
                patient_id=patient_id,
                csv_metadata=csv_metadata,
                image_path=image_path,
                per_image_count=per_image_count,
                size=target_size,
                bg_pixels_max_ratio=0.4,
                start_index=start_index,
                rng=rng
            )
            start_index = idx
            metadata.extend(metapoints)
        else:
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
                    only_masses=only_masses,
                    allowed_calcifications_birads_values=allowed_calcifications_birads_values
                )
                start_index = idx
                # Add the metapoint objects of each contour to our metadata list
                metadata.extend(lesion_metapoints)
    return metadata

def create_bcdr_metadata(
    subdirectories_list: List[str], 
    healthy: bool = False,
    per_image_count: int = 5, 
    target_size: int = 128,
    rng = np.random.default_rng(),
    only_masses: bool = False
) -> List[dict]:
    metadata = []
    if healthy:
        start_index = 0
        for subdirectory_symbol, subdirectory_name in BCDR_HEALTHY_SUBDIRECTORIES.items():
            csv_path = BCDR_ROOT_PATH / subdirectory_name / f"bcdr_{subdirectory_symbol}_img.csv"
            img_csv_df = pd.read_csv(csv_path)
            img_dir_path = BCDR_ROOT_PATH / subdirectory_name
            for _, row in tqdm(img_csv_df.iterrows()):
                metapoints, start_index = generate_healthy_bcdr_metapoints(
                    image_dir_path=img_dir_path,
                    row_df=row,
                    per_image_count=per_image_count,
                    size=target_size,
                    start_index=start_index,
                    bg_pixels_max_ratio=0.4,
                    rng=rng
                )
                metadata.extend(metapoints)
    else:
        assert all(subdir in BCDR_SUBDIRECTORIES.keys() for subdir in subdirectories_list), "Unknown subdirectory symbol"
        for subdirectory in subdirectories_list:
            csv_outlines_path = BCDR_ROOT_PATH / BCDR_SUBDIRECTORIES[subdirectory] / f"bcdr_{subdirectory}_outlines.csv"
            outlines_df = pd.read_csv(csv_outlines_path)
            img_dir_path = BCDR_ROOT_PATH / BCDR_SUBDIRECTORIES[subdirectory]
            for _, row in outlines_df.iterrows():
                lesion_metapoint = generate_bcdr_metapoints(image_dir_path=img_dir_path, row_df=row)
                if only_masses and ('nodule' not in lesion_metapoint['roi_type']): # only keep nodules (i.e. masses)
                    continue
                else:
                    metadata.append(lesion_metapoint)
    return metadata

def create_cbis_ddsm_metadata(
    subsets: List[str],
    healthy: bool = False, 
    per_image_count: int = 5, 
    target_size: int = 128,
    rng = np.random.default_rng(),
    only_masses: bool = False,
    allowed_calcifications_birads_values: List = []
) -> List[dict]:

    metadata = []
    csv_paths = []
    # Select csv files with train/test and mass/calcification samples
    for subset in subsets:
        csv_paths.append(CBIS_DDSM_ROOT_PATH / CBIS_DDSM_CSV_DICT[subset+"_mass"])
        if not only_masses:
            csv_paths.append(CBIS_DDSM_ROOT_PATH / CBIS_DDSM_CSV_DICT[subset+"_calc"])

    for csv_path in csv_paths:
        cbis_ddsm_df = read_csv(csv_path, ",")

        # Corrections for consistency in metadata.json file
        cbis_ddsm_df = cbis_ddsm_df.replace('mass','Mass')
        cbis_ddsm_df = cbis_ddsm_df.replace('calcification','Calcification')
        cbis_ddsm_df = cbis_ddsm_df.replace('MALIGNANT','Malign')
        cbis_ddsm_df = cbis_ddsm_df.replace('BENIGN','Benign')
        cbis_ddsm_df = cbis_ddsm_df.replace('BENIGN_WITHOUT_CALLBACK','Benign_without_callback')
        cbis_ddsm_df = cbis_ddsm_df.replace('RIGHT','R')
        cbis_ddsm_df = cbis_ddsm_df.replace('LEFT','L')
        cbis_ddsm_df = cbis_ddsm_df.replace(r'\n','', regex=True)
        cbis_ddsm_df = cbis_ddsm_df.rename(columns={'breast_density': 'breast density'})

        for index, sample in tqdm(cbis_ddsm_df.iterrows(), total=cbis_ddsm_df.shape[0]):
            image_path = CBIS_DDSM_ROOT_PATH / sample['image file path'].replace("000000.dcm", "1-1.dcm")
            image_id = int(image_path.parent.suffix.replace('.',''))
            if image_path.is_file():
                if healthy:
                    raise("No healthy samples in CBIS-DDSM dataset.")
                else:
                    ds = dicom.dcmread(image_path)
                    mask_path = None
                    mask_directory = CBIS_DDSM_ROOT_PATH / Path(sample['ROI mask file path']).parent
                    for filename in os.listdir(mask_directory):
                        file_path = mask_directory / filename
                        ds_mask = dicom.dcmread(file_path)
                        # Patches and masks can have the same name, so we need to distinguish them by size threshold
                        if ds_mask.pixel_array.shape[0]>ds.pixel_array.shape[0]*0.5:
                            if ds_mask.pixel_array.shape == ds.pixel_array.shape:
                                mask_path = file_path
                                mask = ds_mask.pixel_array
                            else:
                                # Mask size mismatch. Rescaling applied.
                                mask = cv2.resize(ds_mask.pixel_array,list(ds.pixel_array.shape)[::-1], interpolation=cv2.INTER_NEAREST)
                            mask_path = file_path
                    if not mask_path:
                        mask = np.zeros(ds.pixel_array.shape)
                        print(f'No mask found. Please review why. Path: {mask_directory}')

                    lesion_metapoints = generate_cbis_ddsm_metapoints(
                        mask=mask, image_id=image_id,
                        patient_id=sample['patient_id'], csv_metadata=sample,
                        image_path=image_path,
                        roi_type=sample['abnormality type'],
                        start_index=0,
                        only_masses=only_masses,
                        allowed_calcifications_birads_values=allowed_calcifications_birads_values,
                        mask_path=mask_path
                    )
                    # Add the metapoint objects of each contour to our metadata list
                    metadata.extend(lesion_metapoints)
    return metadata

if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed) # seed the random generator
    metadata = []
    if "inbreast" in args.dataset:
        metadata.extend(create_inbreast_metadata(
            healthy=args.healthy, 
            per_image_count=args.per_image_count, 
            target_size=args.healthy_size,
            rng=rng, 
            only_masses=args.only_masses,
            allowed_calcifications_birads_values=args.allowed_calcifications_birads_values
        ))
    if "bcdr" in args.dataset:
        metadata.extend(create_bcdr_metadata(
            args.bcdr_subfolder, 
            healthy=args.healthy,
            per_image_count=args.per_image_count, 
            target_size=args.healthy_size,
            rng=rng,
            only_masses=args.only_masses
        ))
    if "cbis-ddsm" in args.dataset:
        metadata.extend(create_cbis_ddsm_metadata(
            args.cbis_ddsm_csv_set, 
            healthy=args.healthy,
            per_image_count=args.per_image_count, 
            target_size=args.healthy_size,
            rng=rng,
            only_masses=args.only_masses,
            allowed_calcifications_birads_values=args.allowed_calcifications_birads_values
        ))
    # Output metadata as json file to specified location on disk
    outpath = Path(args.output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath.resolve(), "w") as outfile:
        json.dump(metadata, outfile, indent=4)
    print(f"Saved {len(metadata)} metapoints to {outpath.resolve()}")
