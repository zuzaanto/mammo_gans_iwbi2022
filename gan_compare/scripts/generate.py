import argparse
from dataclasses import asdict
from pathlib import Path
from time import time

import cv2
from dacite import from_dict

from gan_compare.data_utils.utils import interval_mapping
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
        "--model_checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint .pt file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="How many samples to generate",
    )
    parser.add_argument(
        "--dont_show_images",
        action="store_true",
        help="Whether to show the generated images in UI.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Whether to save the generated images.",
    )
    parser.add_argument(
        "--out_images_path",
        type=str,
        default=None,
        help="Directory to save the generated images in.",
    )
    parser.add_argument(
        "--birads", type=int, default=None,
        help="Define the associated risk of malignancy (1-6) accroding to the Breast Imaging-Reporting and Data "
             "System (BIRADS).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Load model and config
    assert (
        Path(args.model_checkpoint_dir).is_dir()
    ), f'The path to the model dir you provided does not point to a valid dir:{args.model_checkpoint_dir} '
    yaml_path = next(Path(args.model_checkpoint_dir).rglob("*.yaml"))  # i.e. "config.yaml")
    config_dict = load_yaml(path=yaml_path)
    config = from_dict(GANConfig, config_dict)
    print(asdict(config))
    print("Loading model...")
    model = GANModel(
        model_name=args.model_name,
        config=config,
        dataloader=None,
    )

    if config.conditional is False and args.birads is not None:
        print(
            f'You want to generate ROIs with birads={args.birads}. Note that the GAN model you provided is not '
            f'conditioned on BIRADS. Therefore, it will generate unconditional random samples.')
        args.birads = None
    elif config.conditional is True and args.birads is not None:
        print(f'Conditional samples will be generate for BIRADS = {args.birads}.')

    if args.model_checkpoint_path is None:
        try:
            args.model_checkpoint_path = next(Path(args.model_checkpoint_dir).rglob("*model.pt"))  # i.e. "model.pt"
        except StopIteration:
            try:
                # As there is no model.pt file, let's try to get the last item of the iterator instead, i.e. "300.pt"
                *_, args.model_checkpoint_path = Path(args.model_checkpoint_dir).rglob("*.pt")
            except ValueError:
                pass

    # Let's validate the path to the model
    assert (
            args.model_checkpoint_path is not None and Path(args.model_checkpoint_path).is_file()
    ), f'There seems to be no model file with extension .pt stored in the model_checkpoint_dir you provided: {args.model_checkpoint_dir}'

    print(f'Using model retrieved from: {args.model_checkpoint_path}')

    img_list = model.generate(
        model_checkpoint_path=args.model_checkpoint_path, num_samples=args.num_samples, fixed_condition=args.birads
    )

    # Show the images in interactive UI
    if args.dont_show_images is False:
        for img_ in img_list:
            img_ = interval_mapping(img_.transpose(1, 2, 0), -1.0, 1.0, 0, 255)
            img_ = img_.astype("uint8")
            cv2.imshow("sample", img_ * 2)
            k = cv2.waitKey()
            if k == 27 or k == 32:  # Esc key or space to stop
                break
        cv2.destroyAllWindows()

    # Save the images to model checkpoint folder
    if args.out_images_path is None and args.save_images:
        args.out_images_path = Path(args.model_checkpoint_dir)

    if args.save_images:
        for i, img_ in enumerate(img_list):
            img_path = args.out_images_path / f"{args.model_name}_{i}_{time()}.png"
            img_ = interval_mapping(img_.transpose(1, 2, 0), -1.0, 1.0, 0, 255)
            img_ = img_.astype("uint8")
            cv2.imwrite(str(img_path.resolve()), img_)
        print(f"Saved generated images to {args.out_images_path.resolve()}")
