from __future__ import print_function

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import torchvision.transforms as transforms
from dacite import from_dict
from torch.utils.data import DataLoader

from gan_compare.training.dataset import InbreastDataset
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.gan_model import GANModel
from gan_compare.training.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name: supported: dcgan and lsgan",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="gan_compare/configs/dcgan_config.yaml",
        help="Path to a yaml model config file",
    )
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="Whether to save the dataset samples.",
    )
    parser.add_argument(
        "--out_dataset_path",
        type=str,
        default="visualisation/inbreast_dataset/",
        help="Directory to save the dataset samples in.",
    )
    parser.add_argument(
        "--in_metadata_path",
        type=str,
        default="metadata/metadata.json",
        help="File system location of metadata.json file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Parse config file
    config_dict = load_yaml(path=args.config_path)
    config = from_dict(GANConfig, config_dict)
    print(asdict(config))

    # When we have more datasets implemented, we can specify which one(s) to load in config.
    if config.is_training_data_augmented:
        inbreast_dataset = InbreastDataset(
            metadata_path=args.in_metadata_path,
            final_shape=(config.image_size, config.image_size),
            conditional_birads=config.conditional,
            is_trained_on_masses=config.is_trained_on_masses,
            is_trained_on_calcifications=config.is_trained_on_calcifications,
            is_trained_on_other_roi_types=config.is_trained_on_other_roi_types,
            is_condition_binary=config.is_condition_binary,
            # https://pytorch.org/vision/stable/transforms.html
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # scale: min 0.75 of original image pixels should be in crop, radio: randomly between 3:4 and 4:5
                transforms.RandomResizedCrop(size=config.image_size, scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333)),
                # RandomAffine is not used to avoid edges with filled pixel values to avoid that the generator learns this bias
                # which is not present in the original images.
                #transforms.RandomAffine(),
            ]))
    else:
        inbreast_dataset = InbreastDataset(
            metadata_path=args.in_metadata_path,
            final_shape=(config.image_size, config.image_size),
            conditional_birads=config.conditional,
            is_trained_on_masses=config.is_trained_on_masses,
            is_trained_on_calcifications=config.is_trained_on_calcifications,
            is_trained_on_other_roi_types=config.is_trained_on_other_roi_types,
        )
    print(f"Loaded dataset {inbreast_dataset.__class__.__name__}, with augmentations(?): {config.is_training_data_augmented}")


    dataloader = DataLoader(
        inbreast_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )

    if args.save_dataset:
        output_dataset_dir = Path(args.out_dataset_path)
        if not output_dataset_dir.exists():
            os.makedirs(output_dataset_dir.resolve())
        for i in range(len(inbreast_dataset)):
            # print(inbreast_dataset[i])
            # Plot some training images
            if config.conditional:
                image, condition = inbreast_dataset.__getitem__(i, to_save=True)
                cv2.imwrite(
                    str(output_dataset_dir / f"{i}_birads{condition}.png"),
                    image,
                )
            else:
                cv2.imwrite(
                    str(output_dataset_dir / f"{i}.png"),
                    inbreast_dataset.__getitem__(i, to_save=True),
                )

    print("Loading model...")
    model = GANModel(
        model_name=args.model_name,
        config=config,
        dataloader=dataloader,
        out_dataset_path=args.out_dataset_path,
    )
    print("Loaded model. Starting training...")
    model.train()
