import glob
import io
import json

import plistlib
from pathlib import Path
from typing import Tuple, Union, List
import json

import cv2
import numpy as np
import pandas as pd
from skimage.draw import polygon
import random
import pydicom as dicom
from deprecation import deprecated
from gan_compare.paths import INBREAST_IMAGE_PATH
from gan_compare.dataset.constants import BCDR_VIEW_DICT

import logging


def load_inbreast_mask(
        mask_file: io.BytesIO, imshape: Tuple[int, int] = (4084, 3328),
        expected_roi_type: str = None
) -> np.ndarray:
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
    #mask = np.zeros(imshape)
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

        # Define the ROI types that we want marked as masses, map to lowercase
        roi_type_mass_definition_list = list(map(lambda x:x.lower(), ['Mass', 'Spiculated Region', 'Espiculated Region', 'Spiculated region']))

        # Define the ROI types that we want marked as calcifications, map to lowercase
        roi_type_calc_definition_list = list(map(lambda x:x.lower(),['Calcification', 'Calcifications', 'Cluster']))

        points = roi["Point_px"]
        assert numPoints == len(points)
        points = [eval(point) for point in points]
        if len(points) <= 2:
            for point in points:
                if roi_type.lower() in roi_type_mass_definition_list:
                    mask_masses[int(point[1]), int(point[0])] = 1
                elif roi_type.lower() in roi_type_calc_definition_list:
                    mask_calcifications[int(point[1]), int(point[0])] = 1
                else:
                    mask_other[int(point[1]), int(point[0])] = 1
                    # logging.info(f"Neither Mass nor Calcification, but rather '{roi_type}'. Will be treated as roi type "
                    # f"'Other'. Please consider including '{roi_type}' as dedicated roi_type.")
        else:
            x, y = zip(*points)
            col, row = np.array(x), np.array(
                y
            )
            # x coord is the column coord in an image and y is the row
            poly_x, poly_y = polygon(row, col, shape=imshape)
            if roi_type.lower() in roi_type_mass_definition_list:
                mask_masses[poly_x, poly_y] = 1
                #mask[poly_x, poly_y] = 1
            elif roi_type.lower() in roi_type_calc_definition_list:
                mask_calcifications[poly_x, poly_y] = 1
                #mask[poly_x, poly_y] = 1
            else:
                mask_other[poly_x, poly_y] = 1
                #mask[poly_x, poly_y] = 1
                # logging.info(f"Neither Mass nor Calcification, but rather '{roi_type}'. Will be treated as roi_type "
                # f"'Other'. Please consider including '{roi_type}' as dedicated roi_type.")

    # TODO I don't see the reason for creating dictionaries here, especially that they're not handled later. Ideas @Richard?
    # If a specific expected roi type was provided, only return those. Else, return all possible pre-defined roi types.
    if expected_roi_type is None or expected_roi_type.lower() in roi_type_mass_definition_list:
        mask_list.append({'mask': mask_masses, 'roi_type': 'Mass'})
    if expected_roi_type is None or expected_roi_type.lower() in roi_type_calc_definition_list:
        mask_list.append({'mask': mask_calcifications, 'roi_type': 'Calcification'})
    if expected_roi_type is None or expected_roi_type == 'Other':
        mask_list.append({'mask': mask_other, 'roi_type': 'Other'})

    return mask_list
    # return mask


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

@deprecated()
def is_malignant_estimation_basing_on_birads(birads: str) -> bool:
    """
    Note that this is just an assumption basing on radiologist's opinion, unconfirmed by actual biopsy!
    Currently BIRADS scores up to 4a are considered benign, anything above is considered malignant.
    """
    if int(birads) == 0:
        return None
    if int(birads) < 4:
        return False
    if int(birads) > 4:
        return True
    if birads == "4a":
        return False
    return True


def generate_inbreast_metapoints(
    mask, 
    image_id, 
    patient_id, 
    csv_metadata, 
    image_path, 
    xml_filepath, 
    roi_type: str = "undefined",
    start_index: int = 0,
    only_masses: bool = False,
    allowed_calcifications_birads_values = []
) -> Tuple[list, int]:
    # transform mask to a contiguous np array to allow its usage in C/Cython. mask.flags['C_CONTIGUOUS'] == True?
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    lesion_metapoints = []
    # For each contour, generate a metapoint object including the bounding box as rectangle
    for indx, c in enumerate(contours):
        if (allowed_calcifications_birads_values is not None) and (csv_metadata["Bi-Rads"] not in allowed_calcifications_birads_values): # don't add if it's not an allowed birads value
            continue
        elif only_masses and ('Mass' != str(roi_type)): # don't add if it's not a mass
            continue
        elif c.shape[0] < 2:
            continue
        else:
            # TODO this should be replaced with a Metapoint class object!
            metapoint = {
                "image_id": image_id,
                "patient_id": patient_id,
                "density": int(csv_metadata["ACR"]),
                "birads": csv_metadata["Bi-Rads"],
                "laterality": csv_metadata["Laterality"],
                "view": csv_metadata["View"],
                "lesion_id": csv_metadata["View"] + "_" + str(start_index),
                "bbox": cv2.boundingRect(c),
                "image_path": str(image_path.resolve()),
                "xml_path": str(xml_filepath.resolve()),
                "roi_type": str(roi_type),
                "biopsy_proven_status": None,
                "dataset": "inbreast",
                # "contour": c.tolist(),
            }
            start_index += 1
            # logging.info(f' patent = {patient_id}, start_index = {start_index}')
            lesion_metapoints.append(metapoint)
    return lesion_metapoints, start_index


def _random_crop(image: np.ndarray, size: int, rng) -> Tuple[np.ndarray, List[int]]:
    height, width = image.shape
    ys = int(rng.integers(0, height - size + 1))
    xs = int(rng.integers(0, width - size + 1))
    image_crop = image[ys:ys+size, xs:xs+size]
    return image_crop, [ys, xs, size, size]


def generate_healthy_inbreast_metapoints(
    image_id, 
    patient_id, 
    csv_metadata, 
    image_path, 
    per_image_count: int,
    size: int,
    bg_pixels_max_ratio: float = 0.4,
    start_index: int = 0,
    rng = np.random.default_rng()
) -> Tuple[list, int]:
    lesion_metapoints = []
    if int(csv_metadata["Bi-Rads"][:1]) == 1:
        for _ in range(per_image_count):
            img = convert_to_uint8(dicom.dcmread(image_path).pixel_array)
            img_crop = np.zeros(shape=(size, size))
            thres = 10
            bin_img_crop = img_crop.copy()
            while cv2.countNonZero(bin_img_crop) < (1 - bg_pixels_max_ratio) * size * size:
                img_crop, bbox = _random_crop(img, size, rng)
                _, bin_img_crop = cv2.threshold(img_crop, thres, 255, cv2.THRESH_BINARY)
            if csv_metadata["ACR"].strip() == "":
                continue
            metapoint = {
                "healthy": True,
                "image_id": image_id,
                "patient_id": patient_id,
                "density": int(csv_metadata["ACR"].strip()),
                "birads": csv_metadata["Bi-Rads"],
                "laterality": csv_metadata["Laterality"],
                "view": csv_metadata["View"],
                "lesion_id": csv_metadata["View"] + "_" + str(start_index),
                "bbox": bbox,
                "image_path": str(image_path.resolve()),
                "xml_path": None,
                "roi_type": "healthy",
                "biopsy_proven_status": None,
                "dataset": "inbreast",
            }
            start_index += 1
            lesion_metapoints.append(metapoint)
    return lesion_metapoints, start_index


def generate_bcdr_metapoints(
    image_dir_path: Path, 
    row_df: pd.Series,
):
    laterality, view = get_bcdr_laterality_and_view(row_df)
    if row_df["image_filename"][0] == " ":
        row_df["image_filename"] = row_df["image_filename"][1:]
    if type(row_df["density"]) == str:
        if row_df["density"].strip().startswith("N"):
            density = None
        else:
            density = int(row_df["density"].strip())
    else:
        density = row_df["density"]
    metapoint = {
        "image_id": row_df["study_id"],
        "patient_id": row_df["patient_id"],
        "density": density,
        "birads": None,
        "laterality": laterality,
        "view": view,
        "lesion_id": str(row_df["patient_id"]) + "_" + str(row_df["study_id"]) + "_" + str(row_df["lesion_id"]),
        "bbox": get_bcdr_bbox(row_df["lw_x_points"], row_df["lw_y_points"]),
        "image_path": str((image_dir_path / row_df["image_filename"]).resolve()),
        "xml_path": None,
        "roi_type": get_bcdr_lesion_type(row_df),
        "biopsy_proven_status": row_df["classification"].strip(),
        "dataset": "bcdr",
        # "contour": [parse_str_to_list_of_ints(row_df["lw_x_points"]), parse_str_to_list_of_ints(row_df["lw_y_points"])],
        "contour": None,
    }
    return metapoint


def generate_healthy_bcdr_metapoints(
    image_dir_path: Path, 
    row_df: pd.Series,
    per_image_count: int,
    size: int,
    start_index: int,
    bg_pixels_max_ratio: float = 0.4,
    rng = np.random.default_rng()
):
    laterality, view = get_bcdr_laterality_and_view(row_df, healthy=True)
    if row_df["image_filename"][0] == " ":
        row_df["image_filename"] = row_df["image_filename"][1:]
    image_path = str((image_dir_path / row_df["image_filename"]).resolve())
    metapoints = []
    for _ in range(per_image_count):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_crop = np.zeros(shape=(size, size))
        bin_img_crop = img_crop.copy()
        thres = 10
        while cv2.countNonZero(bin_img_crop) < (1 - bg_pixels_max_ratio) * size * size:
            img_crop, bbox = _random_crop(img, size, rng)
            _, bin_img_crop = cv2.threshold(img_crop, thres, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(bin_img_crop) < 128*12:
            logging.info(str(cv2.countNonZero(bin_img_crop)))

        metapoint = {
            "healthy": True,
            "image_id": row_df["study_id"],
            "patient_id": row_df["patient_id"],
            "density": int(row_df["density"]),
            "birads": None,
            "laterality": laterality,
            "view": view,
            "lesion_id": f"{row_df['study_id']}_{start_index}",
            "bbox": bbox,
            "image_path": image_path,
            "xml_path": None,
            "roi_type": "healthy",
            "biopsy_proven_status": None,
            "dataset": "bcdr",
            "contour": None,
        }
        metapoints.append(metapoint)
        start_index += 1

        # os.makedirs('save_dataset/mymeta', exist_ok=True)
        # cv2.imwrite(f'save_dataset/mymeta/{start_index}.png', np.array(img_crop))

    return metapoints, start_index


def get_bcdr_laterality_and_view(row_df: pd.Series, healthy: bool = False) -> Tuple[str]:
    if healthy:
        view_dict = BCDR_VIEW_DICT[row_df["image_type_id"] + 1]
    else:
        view_dict = BCDR_VIEW_DICT[row_df["image_view"]]
    return view_dict["laterality"], view_dict["view"]


def get_bcdr_lesion_type(case: pd.Series) -> List[str]:
    pathologies = []
    if int(case['mammography_nodule']):
        pathologies.append('nodule')
    if int(case['mammography_calcification']):
        pathologies.append('calcification')
    if int(case['mammography_microcalcification']):
        pathologies.append('microcalcification')
    if int(case['mammography_axillary_adenopathy']):
        pathologies.append('axillary_adenopathy')
    if int(case['mammography_architectural_distortion']):
        pathologies.append('architectural_distortion')
    if int(case['mammography_stroma_distortion']):
        pathologies.append('stroma_distortion')
    return pathologies


def parse_str_to_list_of_ints(points: str, separator: str = " ") -> List[int]:
    points_list = points.split(separator)[1:]
    return [int(x) for x in points_list]


def get_bcdr_bbox(lw_x_points: List[int], lw_y_points: List[int]) -> List[int]:
    lw_y_points, lw_x_points = parse_str_to_list_of_ints(lw_y_points), parse_str_to_list_of_ints(lw_x_points)
    tl_x, tl_y = min(lw_x_points), min(lw_y_points)
    width, height = max(lw_x_points) - tl_x, max(lw_y_points) - tl_y
    return [tl_x, tl_y, width, height]


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    # normalize value range between 0 and 255 and convert to 8-bit unsigned integer
    img_n = cv2.normalize(
        src=image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    return img_n


def save_metadata_to_file(metadata_df: pd.DataFrame, out_path: Path) -> None:
    if len(metadata_df) > 0:
        if not out_path.parent.exists():
            os.makedirs(out_path.parent)
        with open(str(out_path.resolve()), "w") as out_file:
            json.dump(list(metadata_df.T.to_dict().values()), out_file, indent=4)
            
# TODO REFACTOR
# deprecated
# def shuffle_in_synthetic_metadata(metadata: List[dict], synthetic_metadata_path: str, synthetic_shuffle_proportion: float) -> List[dict]:
#     assert Path(synthetic_metadata_path).is_file(), "Incorrect synthetic metadata path"
#     with open(synthetic_metadata_path, "r") as synth_metadata_file:
#         synthetic_metadata = json.load(synth_metadata_file)
#     num_of_metapoints = len(metadata)
#     num_of_synth_metapoints = round(len(metadata) * synthetic_shuffle_proportion)
#     if num_of_synth_metapoints > len(synthetic_metadata):
#         num_of_synth_metapoints = len(synthetic_metadata)
#         num_of_metapoints = round((1 - synthetic_shuffle_proportion) / synthetic_shuffle_proportion * num_of_synth_metapoints)
#     return random.sample(metadata, num_of_metapoints) + random.sample(synthetic_metadata, num_of_synth_metapoints)
