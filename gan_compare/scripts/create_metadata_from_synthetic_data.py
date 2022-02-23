import argparse
import json
import os.path
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to json file to store metadata in.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to (synthetic) images for which to generate metadata.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="The pixel dimensions of the image (assumption: image is a quadratic rectangle)",
    )
    args = parser.parse_args()
    # if args.model_config_path is None:
    #    args.model_config_path = Path(args.checkpoint_path).parent / "config.yaml"
    return args


if __name__ == "__main__":
    args = parse_args()
    metadata = []
    image_size = args.image_size
    path = args.input_path
    valid_images = [".png"]
    for i, f in enumerate(os.listdir(path)):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_path = os.path.join(path, f)
        metapoint = {
            "image_id": -i,
            "patient_id": "synth",
            "density": "-1",
            "birads": None,
            "laterality": "synth",
            "view": "synth",
            "lesion_id": i,
            "bbox": [0, 0, image_size, image_size],
            "image_path": str(img_path),
            "xml_path": "",
            "dataset": "synthetic",
        }
        metadata.append(metapoint)
    outpath = Path(args.output_path)
    if not outpath.parent.exists():
        os.makedirs(outpath.parent.resolve(), exist_ok=True)
    with open(args.output_path, "w") as outfile:
        json.dump(metadata, outfile, indent=4)
    print(f"Saved {len(metadata)} metapoints to {outpath.resolve()}")
