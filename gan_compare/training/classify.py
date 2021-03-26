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
from gan_compare.training.networks.classification.classifier_64 import Net
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="gan_compare/configs/dcgan_config.yaml",
        help="Path to a yaml model config file",
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
    print(
        "Loading dataset..."
    )  # When we have more datasets implemented, we can specify which one(s) to load in config.

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    inbreast_dataset = InbreastDataset(
        metadata_path=args.in_metadata_path,
        final_shape=(config.image_size, config.image_size),
        conditional_birads=config.conditional,
        transform=transform,
    )
    dataloader = DataLoader(
        inbreast_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )

    net = Net(num_labels=config.n_cond)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("Finished Training")
