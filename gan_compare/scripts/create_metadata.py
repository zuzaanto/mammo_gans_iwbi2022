import argparse
import json

import numpy as np
import pydicom as dicom
from tqdm import tqdm

from gan_compare.data_utils.utils import load_inbreast_mask, get_file_list, read_csv, generate_metapoints
from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH, INBREAST_CSV_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", required=True, help="Path to json file to store metadata in."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
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
            lesion_metapoints, idx = generate_metapoints(mask=mask_dict.get('mask'), image_id=image_id,
                                                         patient_id=patient_id, csv_metadata=csv_metadata,
                                                         image_path=image_path, xml_filepath=xml_filepath,
                                                         roi_type=mask_dict.get('roi_type'),
                                                         start_index=start_index,
                                                         )
            start_index = idx
            # Add the metapoint objects of each contour to our metadata list
            metadata.extend(lesion_metapoints)

    # Output metadata as json file to specified location on disk
    outpath = Path(args.output_path)
    if not outpath.parent.exists():
        os.makedirs(outpath.parent.resolve(), exist_ok=True)
    with open(args.output_path, "w") as outfile:
        json.dump(metadata, outfile, indent=4)
    print(f"Saved {len(metadata)} metapoints to {args.output_path}")
