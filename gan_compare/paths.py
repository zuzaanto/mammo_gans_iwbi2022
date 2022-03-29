from pathlib import Path

INBREAST_ROOT_PATH = Path("/path/to/datasets/InBreast/")
BCDR_ROOT_PATH = Path("/path/to/datasets/BCDR/")
CBIS_DDSM_ROOT_PATH = Path("/path/to/datasets/CBIS-DDSM/")


DATASET_PATH_DICT = {
    "bcdr": BCDR_ROOT_PATH,
    "inbreast": INBREAST_ROOT_PATH,
    "cbis-ddsm": CBIS_DDSM_ROOT_PATH,
}

INBREAST_IMAGE_PATH = INBREAST_ROOT_PATH / "AllDICOMs/"
INBREAST_XML_PATH = INBREAST_ROOT_PATH / "AllXML/"
INBREAST_ROI_PATH = INBREAST_ROOT_PATH / "AllROI/"
INBREAST_CSV_PATH = INBREAST_ROOT_PATH / "INbreast.csv"
