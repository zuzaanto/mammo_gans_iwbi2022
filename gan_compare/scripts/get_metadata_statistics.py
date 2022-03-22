import argparse
import json

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_path", required=True, help="Path to json file with metadata."
    )
    parser.add_argument(
        "--attributes",
        type=str,
        default=[
            "density",
            "birads",
            "laterality",
            "view",
            "biopsy_proven_status",
            "roi_type",
        ],
        nargs="+",
        help="Metadata attributes to compute histograms of.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    metadata = None
    attributes = args.attributes
    with open(args.metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    dataset_stats = {}

    for metapoint in tqdm(metadata):
        dataset = metapoint.dataset
        if dataset not in dataset_stats.keys():
            dataset_stats[dataset] = {}
            for name in attributes:
                dataset_stats[dataset][name] = []
        for key, value in metapoint.items():
            if key in dataset_stats[dataset].keys():
                if value is None:
                    value = "None"
                if type(value) == list:
                    for subvalue in value:
                        dataset_stats[dataset][key].append(subvalue)
                else:
                    dataset_stats[dataset][key].append(value)

    for dataset, feature_samples in dataset_stats.items():
        print(
            dataset.upper()
            + ": "
            + str(len(feature_samples[attributes[0]]))
            + " metapoints"
        )

        fig = plt.figure(constrained_layout=True)
        fig.set_dpi(300)
        fig.suptitle("Attribute histograms of " + dataset.upper())
        mosaic = """
                123
                456
                """
        ax_dict = fig.subplot_mosaic(mosaic)

        for (key, value), ax in zip(feature_samples.items(), ax_dict.values()):
            ax.set_title(key, fontsize=7)
            labels, counts = np.unique(value, return_counts=True)
            ax.bar(labels, counts, align="center")
            ax.set_xticks(labels)
            ax.tick_params(axis="both", which="major", labelsize=7)
            if key == "biopsy_proven_status":
                ax.tick_params(axis="x", which="major", labelsize=4)

        plt.show()
