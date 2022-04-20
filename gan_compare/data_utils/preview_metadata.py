import argparse
import json
from dataclasses import fields
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from dacite import from_dict

from gan_compare.data_utils.utils import (
    get_image_from_metapoint,
    get_mask_from_metapoint,
)
from gan_compare.dataset.metapoint import Metapoint


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
        metadata = [
            from_dict(Metapoint, metapoint) for metapoint in json.load(metadata_file)
        ]
    toggle = True
    for metapoint in metadata:
        image_path = metapoint.image_path
        # if metapoint.dataset in ["cbis-ddsm", "inbreast"]:
        # continue
        image = get_image_from_metapoint(metapoint)
        mask = get_mask_from_metapoint(metapoint)

        image_masked = image * (1 - mask)
        fig = plt.figure()
        fig.set_dpi(300)
        ax = plt.subplot(121)
        ax.axis("off")
        fig.suptitle(
            metapoint.dataset.upper() + " " + str(metapoint.patient_id),
            fontsize=15,
            fontweight="bold",
        )
        info_x = image.shape[1] * 1.2
        info_y = 0
        text = "Metadata: \n"
        for field in fields(metapoint):
            key = field.name
            value = getattr(metapoint, field.name)
            text += key + ": " + "%.24s" % str(value) + "\n"
        ax.text(info_x, info_y, text, fontsize=9, va="top", linespacing=2)

        bbox = metapoint.bbox
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
                "View: " + str(metapoint.laterality) + "_" + str(metapoint.view)
            )
        else:
            display = plt.imshow(image_masked)
            ax.set_title(
                "View: "
                + str(metapoint.laterality)
                + "_"
                + str(metapoint.view)
                + "_MASKED"
            )

        def onclick(event):
            global toggle
            toggle = not toggle
            if toggle:
                display.set_data(image)
                ax.set_title(
                    "View: " + str(metapoint.laterality) + "_" + str(metapoint.view)
                )
            else:
                display.set_data(image_masked)
                ax.set_title(
                    "View: "
                    + str(metapoint.laterality)
                    + "_"
                    + str(metapoint.view)
                    + "_MASKED"
                )
            event.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

        plt.show()
