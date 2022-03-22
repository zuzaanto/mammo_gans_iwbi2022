import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

from gan_compare.data_utils.utils import load_inbreast_mask
from gan_compare.paths import INBREAST_IMAGE_PATH, INBREAST_XML_PATH

if __name__ == "__main__":
    for image_path in glob.glob(INBREAST_IMAGE_PATH + "*.dcm"):
        print(image_path)
        ds = dicom.dcmread(image_path)
        patient_id = Path(image_path).stem.split("_")[0]
        print(patient_id)
        xml_filepath = Path(INBREAST_XML_PATH) / f"{patient_id}.xml"
        if xml_filepath.is_file():
            with open(xml_filepath, "rb") as patient_xml:
                mask = load_inbreast_mask(patient_xml, ds.pixel_array.shape)
        else:
            print("No mask found for this image. Assuming no tumors.")
            mask = np.zeros(ds.pixel_array.shape)

        print(ds.pixel_array.shape)
        plt.imshow(ds.pixel_array * (1 - mask))
        plt.show()
