from __future__ import print_function
import argparse
import os
import random
import argparse
from torch.utils.data import DataLoader
from dataclasses import asdict
from gan_compare.training.dataset import InbreastDataset
from gan_compare.training.io import load_yaml
from dacite import from_dict
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.gan_model import GANModel


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Parse config file
    config_dict = load_yaml(path=args.config_path)
    config = from_dict(GANConfig, config_dict)
    print(asdict(config))

    print(
        "Loading dataset..."
    )  # When we have more datasets implemented, we can specify which one(s) to load in config.
    inbreast_dataset = InbreastDataset(
        metadata_path="metadata/metadata.json",
        final_shape=(config.image_size, config.image_size),
        conditional_birads=config.conditional,
    )
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
            print(inbreast_dataset[i])
            # Plot some training images
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
