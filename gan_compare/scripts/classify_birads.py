import argparse
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader

from gan_compare.constants import DATASET_DICT

from dataclasses import asdict
from gan_compare.training.io import load_yaml
from dacite import from_dict
from gan_compare.training.classifier_config import ClassifierConfig
from torch.utils.data.dataset import ConcatDataset
from gan_compare.dataset.synthetic_dataset import SyntheticDataset
from gan_compare.scripts.metrics import calc_all_scores
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="gan_compare/configs/classification_config.yaml",
        help="Path to a yaml model config file",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Parse config file
    config_dict = load_yaml(path=args.config_path)
    config = from_dict(ClassifierConfig, config_dict)
    print(asdict(config))
    print(
        "Loading dataset..."
    )  # When we have more datasets implemented, we can specify which one(s) to load in config.

    train_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )    
    val_transform = transforms.Compose(
        [
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    train_dataset_list = []
    val_dataset_list = []
    test_dataset_list = []
    for dataset_name in config.dataset_names:
        train_dataset_list.append(
            DATASET_DICT[dataset_name](
            metadata_path=config.train_metadata_path,
            final_shape=(config.image_size, config.image_size),
            classify_binary_healthy=config.classify_binary_healthy,
            conditional_birads=True,
            transform=train_transform,
            # synthetic_metadata_path=config.synthetic_metadata_path,
            # synthetic_shuffle_proportion=config.train_shuffle_proportion,
            )
        )
        val_dataset_list.append(
            DATASET_DICT[dataset_name](
                metadata_path=config.validation_metadata_path,
                final_shape=(config.image_size, config.image_size),
                classify_binary_healthy=config.classify_binary_healthy,
                conditional_birads=True,
                transform=val_transform,
                # synthetic_metadata_path=config.synthetic_metadata_path,
                # synthetic_shuffle_proportion=config.validation_shuffle_proportion,
            )
        )
        test_dataset_list.append(
            DATASET_DICT[dataset_name](
                metadata_path=config.test_metadata_path,
                final_shape=(config.image_size, config.image_size),
                classify_binary_healthy=config.classify_binary_healthy,
                conditional_birads=True,
                transform=val_transform,
            )
        )
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)
    if config.use_synthetic:
        # append synthetic data
        synth_train_images = SyntheticDataset(
            metadata_path=config.synthetic_metadata_path,
            final_shape=(config.image_size, config.image_size),
            classify_binary_healthy=config.classify_binary_healthy,
            conditional_birads=True,
            transform=train_transform,
            shuffle_proportion=config.train_shuffle_proportion,
            current_length=len(train_dataset)
        )
        train_dataset = ConcatDataset([train_dataset, synth_train_images])
        synth_val_images = SyntheticDataset(
            metadata_path=config.synthetic_metadata_path,
            final_shape=(config.image_size, config.image_size),
            classify_binary_healthy=config.classify_binary_healthy,
            conditional_birads=True,
            transform=val_transform,
            shuffle_proportion=config.train_shuffle_proportion,
            current_length=len(val_dataset)
        )
        val_dataset = ConcatDataset([val_dataset, synth_val_images])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    if not Path(config.out_checkpoint_path).parent.exists():
        os.makedirs(Path(config.out_checkpoint_path).parent.resolve(), exist_ok=True)

    if config.image_size == 64: from gan_compare.training.networks.classification.classifier_64 import Net
    elif config.image_size == 128: from gan_compare.training.networks.classification.classifier_128 import Net
    else: raise ValueError("image_size must be either 64 or 128")

    net = Net(num_labels=config.n_cond)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    best_loss = 10000
    for epoch in range(config.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
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
                
        # validate
        val_loss = []
        with torch.no_grad():
            y_true = []
            y_prob_logit = []
            net.eval()
            for i, data in enumerate(val_dataloader, 0):
                images, labels = data
                # print(images.size())
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                val_loss.append(criterion(outputs, labels))
                y_true.append(labels)
                y_prob_logit.append(outputs.data)
            val_loss = np.mean(val_loss)
            if val_loss < best_loss:
                torch.save(net.state_dict(), config.out_checkpoint_path)
            calc_all_scores(torch.cat(y_true), torch.cat(y_prob_logit), val_loss, "Valid", epoch)

    print("Finished Training")
    print("Beginning test...")
    net.load_state_dict(torch.load(config.out_checkpoint_path))
    with torch.no_grad():
        y_true = []
        y_prob_logit = []
        test_loss = []
        net.eval()
        for i, data in enumerate(test_dataloader, 0):
            images, labels = data
            # print(images.size())
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_loss.append(criterion(outputs, labels))
            y_true.append(labels)
            y_prob_logit.append(outputs.data)
        test_loss = np.mean(test_loss)
        calc_all_scores(torch.cat(y_true), torch.cat(y_prob_logit), test_loss, "Test")
        # output_ROC_curve(y_true, y_prob_logit, "Test")
    print("Finished testing.")
    print(f"Saved model state dict to {config.out_checkpoint_path}")
