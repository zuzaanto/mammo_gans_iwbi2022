from __future__ import print_function
import argparse
import argparse
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import asdict
from gan_compare.training.dataset import InbreastDataset
from gan_compare.training.io import load_yaml
from dacite import from_dict
from gan_compare.training.classifier_config import ClassifierConfig
from gan_compare.training.networks.classification.classifier_64 import Net
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch


def parse_args() -> argparse.Namespace:
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

    train_inbreast_dataset = InbreastDataset(
        metadata_path=config.train_metadata_path,
        final_shape=(config.image_size, config.image_size),
        conditional_birads=True,
        transform=train_transform,
        synthetic_metadata_path=config.synthetic_metadata_path,
        synthetic_shuffle_proportion=config.train_shuffle_proportion,
    )
    train_dataloader = DataLoader(
        train_inbreast_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )    
    val_inbreast_dataset = InbreastDataset(
        metadata_path=config.validation_metadata_path,
        final_shape=(config.image_size, config.image_size),
        conditional_birads=True,
        transform=val_transform,
        synthetic_metadata_path=config.synthetic_metadata_path,
        synthetic_shuffle_proportion=config.validation_shuffle_proportion,
    )
    val_dataloader = DataLoader(
        val_inbreast_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    test_inbreast_dataset = InbreastDataset(
        metadata_path=config.test_metadata_path,
        final_shape=(config.image_size, config.image_size),
        conditional_birads=True,
        transform=val_transform,
    )
    test_dataloader = DataLoader(
        test_inbreast_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
    )
    if not Path(config.out_checkpoint_path).parent.exists():
        os.makedirs(Path(config.out_checkpoint_path).parent.resolve(), exist_ok=True)

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
        total = 0
        correct = 0
        val_loss = []
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(val_dataloader, 0):
                images, labels = data
                # print(images.size())
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss.append(criterion(outputs, labels))
            val_loss = np.mean(val_loss)
            if val_loss < best_loss:
                torch.save(net.state_dict(), config.out_checkpoint_path)
            print(f'Accuracy in {epoch + 1} epoch: {100 * correct / total}')            
            print(f'Loss in {epoch + 1} epoch: {val_loss}')            

    print("Finished Training")
    print("Beginning test...")
    total = 0
    correct = 0
    test_loss = []
    net.load_state_dict(torch.load(config.out_checkpoint_path))
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_dataloader, 0):
            images, labels = data
            # print(images.size())
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss.append(criterion(outputs, labels))
        test_loss = np.mean(test_loss)
        print(f'Test accuracy: {100 * correct / total}')            
        print(f'Test loss: {test_loss}')            
    print("Finished testing.")
    print(f"Saved model state dict to {config.out_checkpoint_path}")
