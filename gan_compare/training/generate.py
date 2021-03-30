import torch.optim as optim
import argparse
import cv2
import torch
from pathlib import Path
from gan_compare.data_utils.utils import interval_mapping
from gan_compare.training.gan_model import GANModel
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.io import load_yaml
from dacite import from_dict
from dataclasses import asdict
from time import time



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
        "--num_samples", type=int, default=50, help="How many samples to generate"
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Load model and config
    yaml_path = next(Path(args.model_checkpoint_dir).rglob("*.yaml"))  # i.e. "config.yaml")
    config_dict = load_yaml(path=yaml_path)
    config = from_dict(GANConfig, config_dict)
    print(asdict(config))
    print("Loading model...")
    model = GANModel(
        model_name=args.model_name,
        config=config,
        dataloader=None
    )

    if args.model_checkpoint_path is None:
        args.model_checkpoint_path = next(Path(args.model_checkpoint_dir).rglob("*.pt"))  # i.e. "model.pt"

    img_list = model.generate(
        model_checkpoint_path=args.model_checkpoint_path, num_samples=args.num_samples
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
            cv2.imwrite(str(args.out_images_path.resolve()), img_)
        print(f"Saved generated images to {args.out_images_path.resolve()}")

