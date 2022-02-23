import argparse
import json
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

from gan_compare.data_utils.utils import load_inbreast_mask
from gan_compare.dataset.constants import (BCDR_BIRADS_DICT, BIRADS_DICT,
                                           DENSITY_DICT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path", required=True, help="Path to json file with metadata."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    metadata = None
    metadata_path = args.metadata_path
    assert Path(metadata_path).is_file(), f"Metadata not found in {metadata_path}"
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    toggle = True
    for metapoint in metadata:
        image_path = metapoint["image_path"]
        if metapoint["dataset"] in ["cbis-ddsm", "inbreast"]:
            # continue
            ds = dicom.dcmread(image_path)
            image = ds.pixel_array
            if "mask_path" in metapoint:
                if Path(metapoint["mask_path"]).is_file():
                    mask = dicom.dcmread(metapoint["mask_path"]).pixel_array
                    mask = cv2.resize(
                        mask,
                        list(ds.pixel_array.shape)[::-1],
                        interpolation=cv2.INTER_NEAREST,
                    )
            elif "xml_path" in metapoint:
                if Path(metapoint["xml_path"]).is_file():
                    with open(metapoint["xml_path"], "rb") as patient_xml:
                        mask_list = load_inbreast_mask(
                            patient_xml, ds.pixel_array.shape
                        )
                        mask = np.zeros(ds.pixel_array.shape)
                        for mask_sample in mask_list:
                            mask = np.logical_or(mask, mask_sample["mask"])
            else:
                mask = np.zeros(ds.pixel_array.shape)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = np.zeros(image.shape)

        image_masked = image * (1 - mask)
        fig = plt.figure()
        fig.set_dpi(300)
        ax = plt.subplot(121)
        ax.axis("off")
        fig.suptitle(
            metapoint["dataset"].upper() + " " + str(metapoint["patient_id"]),
            fontsize=15,
            fontweight="bold",
        )
        info_x = image.shape[1] * 1.2
        info_y = 0
        text = "Metadata: \n"
        for key, value in metapoint.items():
            text += key + ": " + "%.24s" % str(value) + "\n"
        ax.text(info_x, info_y, text, fontsize=9, va="top", linespacing=2)

        bbox = metapoint["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        if toggle:
            display = plt.imshow(image)
            ax.set_title(
                "View: " + str(metapoint["laterality"]) + "_" + str(metapoint["view"])
            )
        else:
            display = plt.imshow(image_masked)
            ax.set_title(
                "View: "
                + str(metapoint["laterality"])
                + "_"
                + str(metapoint["view"])
                + "_MASKED"
            )

        def onclick(event):
            global toggle
            toggle = not toggle
            if toggle:
                display.set_data(image)
                ax.set_title(
                    "View: "
                    + str(metapoint["laterality"])
                    + "_"
                    + str(metapoint["view"])
                )
            else:
                display.set_data(image_masked)
                ax.set_title(
                    "View: "
                    + str(metapoint["laterality"])
                    + "_"
                    + str(metapoint["view"])
                    + "_MASKED"
                )
            event.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

        plt.show()
