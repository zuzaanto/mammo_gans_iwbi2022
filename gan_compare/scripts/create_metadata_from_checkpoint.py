import argparse
import glob
import json
import os.path
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
from dacite import from_dict
from tqdm import tqdm

from gan_compare.data_utils.utils import (
    get_file_list,
    interval_mapping,
    load_inbreast_mask,
    read_csv,
)
from gan_compare.dataset.constants import BIRADS_DICT
from gan_compare.paths import INBREAST_CSV_PATH, INBREAST_IMAGE_PATH, INBREAST_XML_PATH
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.gan_model import GANModel
from gan_compare.training.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to json file to store metadata in.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model's checkpoint."
    )
    parser.add_argument(
        "--generated_data_dir",
        type=str,
        required=True,
        help="Directory to save generated images in.",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=64,
        help="Number of samples to generate per each class.",
    )
    parser.add_argument(
        "--model_config_path", type=str, default=None, help="Path to model config file."
    )
    parser.add_argument("--gan_type", type=str, default="dcgan", help="Model name.")
    args = parser.parse_args()
    if args.model_config_path is None:
        args.model_config_path = Path(args.checkpoint_path).parent / "config.yaml"
    return args


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_yaml(path=args.model_config_path)
    config = from_dict(GANConfig, config_dict)
    print(asdict(config))
    print("Loading model...")
    model = GANModel(gan_type=args.model_name, config=config, dataloader=None)
    assert (
        config.conditional is True
    ), "Currently creation of synthetic metadata makes sense only for conditional models. Please load a conditional model."

    if config.split_birads_fours:
        birads = BIRADS_DICT.values()
    else:
        birads = list(range(config.birads_min, config.birads_max))
    metadata = []
    for birads_val in tqdm(birads):
        img_list = model.generate(
            args.checkpoint_path,
            fixed_noise=None,
            fixed_condition=birads_val,
            num_samples=args.num_samples_per_class,
        )
        generated_img_birads_dir = Path(args.generated_data_dir) / f"{birads_val}"
        os.makedirs(generated_img_birads_dir.resolve(), exist_ok=True)
        for i, img_ in enumerate(img_list):
            img_path = generated_img_birads_dir / f"{i}.png"
            img_ = interval_mapping(img_.transpose(1, 2, 0), 0.0, 1.0, 0, 255)
            img_ = img_.astype("uint8")
            cv2.imwrite(str(img_path.resolve()), img_)
            metapoint = {
                "image_id": -i,
                "patient_id": "synth",
                "density": "-1",
                "birads": str(birads_val),
                "laterality": "synth",
                "view": "synth",
                "lesion_id": i,
                "bbox": [0, 0, config.image_size, config.image_size],
                "image_path": str(img_path.resolve()),
                "xml_path": "",
                "dataset": "synthetic",
            }
            metadata.append(metapoint)

    print(f"Saved generated images to {Path(args.generated_data_dir).resolve()}")
    outpath = Path(args.output_path)
    if not outpath.parent.exists():
        os.makedirs(outpath.parent.resolve(), exist_ok=True)
    with open(args.output_path, "w") as outfile:
        json.dump(metadata, outfile, indent=4)
    print(f"Saved {len(metadata)} metapoints to {outpath.resolve()}")
