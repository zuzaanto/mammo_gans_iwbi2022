from __future__ import print_function

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import torchvision.transforms as transforms
from dacite import from_dict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from gan_compare.data_utils.utils import collate_fn
from gan_compare.dataset.bcdr_dataset import BCDRDataset
from gan_compare.dataset.inbreast_dataset import InbreastDataset
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
        default="visualisation/dataset/",
        help="Directory to save the dataset samples in.",
    )
    parser.add_argument(
        "--in_metadata_path",
        type=str,
        required=True,
        help="File system location of metadata.json file.",
    )
    args = parser.parse_args()
    return args


DATASET_DICT = {
    "bcdr": BCDRDataset,
    "inbreast": InbreastDataset,
}


if __name__ == "__main__":
    args = parse_args()
    # Parse config file
    config_dict = load_yaml(path=args.config_path)
    config = from_dict(GANConfig, config_dict)
    print(asdict(config))
    dataset_list = []
    transform_to_use = None
    if config.is_training_data_augmented:
        transform_to_use = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # scale: min 0.75 of original image pixels should be in crop, radio: randomly between 3:4 and 4:5
            # transforms.RandomResizedCrop(size=config.image_size, scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333)),
            # RandomAffine is not used to avoid edges with filled pixel values to avoid that the generator learns this bias
            # which is not present in the original images.
            #transforms.RandomAffine(),
        ])
    for dataset_name in config.dataset_names:
        dataset = DATASET_DICT[dataset_name](
            # TODO Remove passing all the config variables one by one. Instead let's only pass the config dict and hhandle its keys internally.
            metadata_path=args.in_metadata_path,
            # https://pytorch.org/vision/stable/transforms.html
            transform=transform_to_use,
            config=config
        )
        dataset_list.append(dataset)
    dataset = ConcatDataset(dataset_list)

    print(f"Loaded dataset {dataset.__class__.__name__}, with augmentations(?): {config.is_training_data_augmented}")

    # drop_last is true to avoid batch_sie of 1 that throws an Value Error in BatchNorm. https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/5
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate_fn,  # Filter out None returned by DataSet.
        drop_last=True,
    )


    if args.save_dataset:
        output_dataset_dir = Path(args.out_dataset_path)
        if not output_dataset_dir.exists():
            os.makedirs(output_dataset_dir.resolve())
        for i in tqdm(range(len(dataset))):
            # print(dataset[i])
            # Plot some training images
            items = dataset.__getitem__(i)
            if items is None:
                continue
            sample, condition, image = items

            out_image_path = f"{i}_{config.conditioned_on}_{condition}.png" if config.conditional else f"{i}.png"
            cv2.imwrite(str(output_dataset_dir / out_image_path), image)
    print("Loading model...")
    model = GANModel(
        model_name=args.model_name,
        config=config,
        dataloader=dataloader,
        out_dataset_path=args.out_dataset_path,
    )
    print("Loaded model. Starting training...")
    model.train()
