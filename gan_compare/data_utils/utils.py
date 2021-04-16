import glob
import io
import plistlib
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
from skimage.draw import polygon

from gan_compare.paths import INBREAST_IMAGE_PATH


def load_inbreast_mask(
        mask_file: io.BytesIO, imshape: Tuple[int, int] = (4084, 3328),
        expected_roi_type:str = None
) -> list:
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_file : Loaded xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    @expected_roi_type : The requested type of lesion in the roi as string e.g. 'Mass' or 'Calcification'
    return: list of dictionaries, each containing the roi type as string and a mask as numpy array where positions
    in the roi are assigned a value of 1. e.g. [{mask:mask1, roi_type:type1}, {mask:mask2, roi_type:type2}]
    """
    mask_list = []
    mask_masses = np.zeros(imshape)
    mask_calcifications = np.zeros(imshape)
    mask_other = np.zeros(imshape)
    plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)["Images"][0]
    numRois = plist_dict["NumberOfROIs"]
    rois = plist_dict["ROIs"]
    assert len(rois) == numRois
    for roi in rois:
        numPoints = roi["NumberOfPoints"]

        # ROI type information i.e. either calcifications or masses
        roi_type = roi["Name"]

        # Define the ROI types that we want marked as masses
        roi_type_mass_definition_list = ['Mass', 'Spiculated Region', 'Espiculated Region', 'Spiculated region']

        # Define the ROI types that we want marked as calcifications
        roi_type_calc_definition_list = ['Calcification', 'Calcifications', 'Cluster']

        points = roi["Point_px"]
        assert numPoints == len(points)
        points = [eval(point) for point in points]
        if len(points) <= 2:
            for point in points:
                if roi_type in roi_type_mass_definition_list:
                    mask_masses[int(point[1]), int(point[0])] = 1
                elif roi_type in roi_type_calc_definition_list:
                    mask_calcifications[int(point[1]), int(point[0])] = 1
                else:
                    mask_other[int(point[1]), int(point[0])] = 1
                    # print(f"Neither Mass nor Calcification, but rather '{roi_type}'. Will be treated as roi type "
                    #      f"'Other'. Please consider including '{roi_type}' as dedicated roi_type.")
        else:
            x, y = zip(*points)
            col, row = np.array(x), np.array(
                y
            )
            # x coord is the column coord in an image and y is the row
            poly_x, poly_y = polygon(row, col, shape=imshape)
            if roi_type in roi_type_mass_definition_list:
                mask_masses[poly_x, poly_y] = 1
            elif roi_type in roi_type_calc_definition_list:
                mask_calcifications[poly_x, poly_y] = 1
            else:
                mask_other[poly_x, poly_y] = 1
                # print(f"Neither Mass nor Calcification, but rather '{roi_type}'. Will be treated as roi_type
                # 'Other'. Please consider including '{roi_type}' as dedicated roi_type.")

    # If a specific expected roi type was provided, only return those. Else, return all possible pre-defined roi types.
    if expected_roi_type is None or expected_roi_type == 'Mass':
        mask_list.append({'mask': mask_masses, 'roi_type': 'Mass'})
    if expected_roi_type is None or expected_roi_type == 'Calcification':
        mask_list.append({'mask': mask_calcifications, 'roi_type': 'Calcification'})
    if expected_roi_type is None or expected_roi_type == 'Other':
        mask_list.append({'mask': mask_other, 'roi_type': 'Other'})

    return mask_list


def get_file_list():
    return glob.glob(str(INBREAST_IMAGE_PATH.resolve()) + "/*.dcm")


def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    # scale to interval [0,1]
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    # multiply by range and add minimum to get interval [min,range+min]
    return to_min + (scaled * to_range)


def read_csv(path: Union[Path, str], sep: chr = ";") -> pd.DataFrame:
    with open(path, "r") as csv_file:
        return pd.read_csv(csv_file, sep=sep)


def generate_metapoints(mask, image_id, patient_id, csv_metadata, image_path, xml_filepath, roi_type: str = "undefined",
                        start_index: int = 0) -> (list, int):
    # transform mask to a contiguous np array to allow its usage in C/Cython. mask.flags['C_CONTIGUOUS'] == True?
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    lesion_metapoints = []
    # For each contour, generate a metapoint object including the bounding box as rectangle
    for indx, c in enumerate(contours):
        if c.shape[0] < 2:
            continue
        metapoint = {
            "image_id": image_id,
            "patient_id": patient_id,
            "ACR": csv_metadata["ACR"],
            "birads": csv_metadata["Bi-Rads"],
            "laterality": csv_metadata["Laterality"],
            "view": csv_metadata["View"],
            "lesion_id": csv_metadata["View"] + "_" + str(start_index),
            "bbox": cv2.boundingRect(c),
            "image_path": str(image_path.resolve()),
            "xml_path": str(xml_filepath.resolve()),
            "roi_type": str(roi_type)
            # "contour": c.tolist(),
        }
        start_index += 1
        # print(f'patent= {patient_id}, start_index = {start_index}')
        lesion_metapoints.append(metapoint)
    return lesion_metapoints, start_index
