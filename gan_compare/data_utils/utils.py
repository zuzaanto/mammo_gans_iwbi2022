from skimage.draw import polygon
from typing import Tuple, Union
import numpy as np
import plistlib
import glob
import io
from pathlib import Path
import pandas as pd
from gan_compare.paths import INBREAST_IMAGE_PATH


def load_inbreast_mask(
    mask_file: io.BytesIO, imshape: Tuple[int, int] = (4084, 3328)
) -> np.ndarray:
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_file : Loaded xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """

    mask = np.zeros(imshape)
    plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)["Images"][0]
    numRois = plist_dict["NumberOfROIs"]
    rois = plist_dict["ROIs"]
    assert len(rois) == numRois
    for roi in rois:
        numPoints = roi["NumberOfPoints"]
        points = roi["Point_px"]
        assert numPoints == len(points)
        points = [eval(point) for point in points]
        if len(points) <= 2:
            for point in points:
                mask[int(point[1]), int(point[0])] = 1
        else:
            x, y = zip(*points)
            col, row = np.array(x), np.array(
                y
            )  ## x coord is the column coord in an image and y is the row
            poly_x, poly_y = polygon(row, col, shape=imshape)
            mask[poly_x, poly_y] = 1
    return mask


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
