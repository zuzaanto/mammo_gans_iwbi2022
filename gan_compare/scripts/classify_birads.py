import argparse
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader

from gan_compare.constants import DATASET_DICT, CLASSIFIERS_DICT

from dataclasses import asdict
from gan_compare.training.io import load_yaml
from dacite import from_dict
from gan_compare.training.classifier_config import ClassifierConfig
from torch.utils.data.dataset import ConcatDataset
from gan_compare.dataset.synthetic_dataset import SyntheticDataset
from gan_compare.scripts.metrics import calc_all_scores, output_ROC_curve
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="gan_compare/configs/classification_config.yaml",
        help="Path to a yaml model config file",
    )
    parser.add_argument(
        "--only_get_metrics",
        action="store_true",
        help="Whether to skip training and just evaluate the model saved in the default location.",
    )
    parser.add_argument(
        "--save_dataset", action="store_true", help="Whether to save image patches to images_classifier dir",
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
            is_trained_on_calcifications=config.is_trained_on_calcifications,
            config=config
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
                is_trained_on_calcifications=config.is_trained_on_calcifications,
                config=config
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
                is_trained_on_calcifications=config.is_trained_on_calcifications,
                config=config
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
            current_length=len(train_dataset),
            config=config
        )
        train_dataset = ConcatDataset([train_dataset, synth_train_images])
        synth_val_images = SyntheticDataset(
            metadata_path=config.synthetic_metadata_path,
            final_shape=(config.image_size, config.image_size),
            classify_binary_healthy=config.classify_binary_healthy,
            conditional_birads=True,
            transform=val_transform,
            shuffle_proportion=config.train_shuffle_proportion,
            current_length=len(val_dataset),
            config=config
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


    device = torch.device(
        "cuda" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu"
    )

    print(f"Device: {device}")

    # net = CLASSIFIERS_DICT[config.model_name](num_classes=config.n_cond, img_size=config.image_size).to(device)
    from gan_compare.training.networks.classification.classifier_128 import Net as Net128
    net = Net128(num_labels=2).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.save_dataset:
        print(f"Saving data samples...")
        save_data_path = Path("save_dataset")

        with open(save_data_path / "validation.txt", 'w') as f:
            f.write('index, y_prob\n')
        cnt = 0
        for data in tqdm(val_dataloader): # this has built-in shuffling; if not shuffled, only lesioned patches will be output first
            samples, labels, images = data
            outputs = net(samples)
            for y_prob_logit, label, image in zip(outputs.data, labels, images):
                y_prob = torch.exp(y_prob_logit)[1]
                with open(save_data_path / "validation.txt", "a") as f:
                    f.write(f'{cnt}, {y_prob}\n')

                label = "healthy" if int(label) == 1 else "with_lesions"
                out_image_dir = save_data_path / "validation" / str(label)
                out_image_dir.mkdir(parents=True, exist_ok=True)
                out_image_path = out_image_dir / f"{cnt}.png" 
                cv2.imwrite(str(out_image_path), np.array(image))

                cnt += 1

                
        print(f"Saved data samples to {save_data_path.resolve()}")

    if not args.only_get_metrics:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        best_loss = 10000
        for epoch in tqdm(range(config.num_epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            print("\nTraining...")
            for i, data in enumerate(tqdm(train_dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                samples, labels, _ = data

                if len(samples) <= 1: continue # batch normalization won't work if samples too small (https://stackoverflow.com/a/48344268/3692004)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(samples.to(device))
                loss = criterion(outputs, labels.to(device))
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
                print("\nValidating...")
                for i, data in enumerate(tqdm(val_dataloader)):
                    samples, labels, _ = data
                    # print(images.size())
                    outputs = net(samples.to(device))
                    val_loss.append(criterion(outputs.cpu(), labels))
                    y_true.append(labels)
                    y_prob_logit.append(outputs.data.cpu())
                val_loss = np.mean(val_loss)
                if val_loss < best_loss:
                    torch.save(net.state_dict(), config.out_checkpoint_path)
                calc_all_scores(torch.cat(y_true), torch.cat(y_prob_logit), val_loss, "Valid", epoch)

        print("Finished Training")
        print(f"Saved model state dict to {config.out_checkpoint_path}")

    print("Beginning test...")
    net.load_state_dict(torch.load(config.out_checkpoint_path))
    with torch.no_grad():
        y_true = []
        y_prob_logit = []
        test_loss = []
        net.eval()
        print("\nTesting...")
        for i, data in enumerate(tqdm(test_dataloader)):
            samples, labels, _ = data
            # print(images.size())
            outputs = net(samples.to(device))
            test_loss.append(criterion(outputs.cpu(), labels))
            y_true.append(labels)
            y_prob_logit.append(outputs.data.cpu())
        test_loss = np.mean(test_loss)
        y_true = torch.cat(y_true)
        y_prob_logit = torch.cat(y_prob_logit)
        calc_all_scores(y_true, y_prob_logit, test_loss, "Test")
        if config.classify_binary_healthy: output_ROC_curve(y_true, y_prob_logit, "Test")
    print("Finished testing.")
    
