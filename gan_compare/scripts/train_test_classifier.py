import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dacite import from_dict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from gan_compare.constants import DATASET_DICT, get_classifier
from gan_compare.data_utils.utils import setup_logger, get_logfilename
from gan_compare.dataset.synthetic_dataset import SyntheticDataset
from gan_compare.scripts.metrics import calc_all_scores, output_ROC_curve, calc_AUROC, calc_AUPRC
from gan_compare.training.classifier_config import ClassifierConfig
from gan_compare.training.io import load_yaml


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
        "--in_checkpoint_path",
        type=str,
        default="model_checkpoints/classifier/classifier.pt",
        help="Only required if --only_get_metrics is set. Path to checkpoint to be loaded.",
    )
    parser.add_argument(
        "--save_dataset", action="store_true", help="Whether to save image patches to images_classifier dir",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    setup_logger()

    args = parse_args()
    # Parse config file
    config_dict = load_yaml(path=args.config_path)
    config = from_dict(ClassifierConfig, config_dict)

    logfilename = get_logfilename(is_time_reset=False)

    config.out_checkpoint_path += f"{logfilename}.pt"

    logging.info(str(asdict(config)))
    logging.info(
        "Loading dataset...")  # When we have more datasets implemented, we can specify which one(s) to load in config.

    if config.use_synthetic:
        assert config.synthetic_data_dir is not None, 'If you want to use synthetic data, you must provide a diretory with the patches in config.synthetic_data_dir.'
    if config.no_transforms:
        train_transform = transforms.Compose(
            [
                transforms.Normalize((0.5), (0.5)),
            ]
        )
    else:
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
                conditional_birads=True,
                transform=train_transform,
                config=config
            )
        )
        val_dataset_list.append(
            DATASET_DICT[dataset_name](
                metadata_path=config.validation_metadata_path,
                conditional_birads=True,
                transform=val_transform,
                config=config
            )
        )
        test_dataset_list.append(
            DATASET_DICT[dataset_name](
                metadata_path=config.test_metadata_path,
                conditional_birads=True,
                transform=val_transform,
                config=config
            )
        )
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)
    if config.use_synthetic:
        # APPEND SYNTHETIC DATA

        synth_train_images = SyntheticDataset(
            conditional_birads=True,
            transform=train_transform,
            shuffle_proportion=config.train_shuffle_proportion,
            current_length=len(train_dataset),
            config=config,
        )
        train_dataset = ConcatDataset([train_dataset, synth_train_images])
        logging.info(f'Number of synthetic patches added to training set: {len(synth_train_images)}')

        # YOU MUST ADD THE HEALTHY PATCHES WHICH ARE MEANT TO BALANCE THE SYNTHETIC PATCHES TO THE TRAINING SET ABOVE
        # TODO REFACTOR
        # # append healthy patches for balancing the synthetic patches (which are all lesioned)
        # train_dataset_list = []
        # for dataset_name in config.dataset_names:
        #     train_dataset_list.append(
        #         DATASET_DICT[dataset_name](
        #         metadata_path=config.synthetic_metadata_path,
        #         conditional_birads=True,
        #         transform=train_transform,
        #         config=config
        #         # synthetic_metadata_path=config.synthetic_metadata_path,
        #         # synthetic_shuffle_proportion=config.train_shuffle_proportion,
        #         )
        #     )
        # train_dataset = ConcatDataset([train_dataset, ConcatDataset(train_dataset_list)])
        # logging.info('Added healthy patches for balancing.')

        # synth_val_images = SyntheticDataset(
        #     metadata_path=config.synthetic_metadata_path,
        #     conditional_birads=True,
        #     transform=val_transform,
        #     shuffle_proportion=config.train_shuffle_proportion,
        #     current_length=len(val_dataset),
        #     config=config,
        #     indices=None
        # )
        # val_dataset = ConcatDataset([val_dataset, synth_val_images])
        # logging.info(f'Number of samples added to synthetic validation set: {len(synth_val_images)}')

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

    logging.info(f"Device: {device}")

    net = get_classifier(name=config.model_name, num_classes=config.n_cond, img_size=config.image_size).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.save_dataset:
        # This code section is only for saving patches as image files and further info about the patch if needed.
        # The program stops execution after this section and performs no training.

        logging.info(f"Saving data samples...")

        # TODO: state_dict name should probably be in config yaml instead of hardcoded.
        net.load_state_dict(torch.load('model_checkpoints/classifier 50 no synth/classifier.pt'))
        net.eval()
        save_data_path = Path("save_dataset")

        with open(save_data_path / "validation.txt", 'w') as f:
            f.write('index, y_prob\n')
        cnt = 0
        metapoints = []
        for data in tqdm(
                test_dataset):  # this has built-in shuffling; if not shuffled, only lesioned patches will be output first
            sample, label, image, r, d = data
            outputs = net(sample[np.newaxis, ...])

            # for y_prob_logit, label, image, r, d in zip(outputs.data, labels, images, rs, ds):
            y_prob_logit = outputs.data
            y_prob = torch.exp(y_prob_logit)
            metapoint = {
                'label': label,
                'roi_type': r,
                'dataset': d,
                'cnt': cnt,
                'y_prob': y_prob.numpy().tolist()
            }
            metapoints.append(metapoint)

            # with open(save_data_path / "validation.txt", "a") as f:
            #     f.write(f'{cnt}, {y_prob}\n')

            label = "healthy" if int(label) == 1 else "with_lesions"
            out_image_dir = save_data_path / "validation" / str(label)
            out_image_dir.mkdir(parents=True, exist_ok=True)
            out_image_path = out_image_dir / f"{cnt}.png"
            cv2.imwrite(str(out_image_path), np.array(image))

            cnt += 1
        with open(save_data_path / 'validation.json', 'w') as f:
            f.write(json.dumps(metapoints))

        logging.info(f"Saved data samples to {save_data_path.resolve()}")
        exit()

    if not args.only_get_metrics:

        # PREPARE TRAINING
        # TODO: Optimizer params (lr, momentum) should be moved to classifier_config.
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        best_loss = 10000
        best_f1 = 0
        best_epoch = 0
        best_prc_auc = 0

        # START TRAINING LOOP
        for epoch in tqdm(range(config.num_epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            logging.info("Training...")
            for i, data in enumerate(tqdm(train_dataloader)):
                # get the inputs; data is a list of [inputs, labels]
                samples, labels, _, _ = data

                if len(
                        samples) <= 1: continue  # batch normalization won't work if samples too small (https://stackoverflow.com/a/48344268/3692004)

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
                    logging.info("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            # VALIDATE

            val_loss = []
            with torch.no_grad():
                y_true = []
                y_prob_logit = []
                net.eval()
                logging.info("Validating...")
                for i, data in enumerate(tqdm(val_dataloader)):
                    samples, labels, _, _ = data
                    # logging.info(images.size())
                    outputs = net(samples.to(device))
                    val_loss.append(criterion(outputs.cpu(), labels))
                    y_true.append(labels)
                    y_prob_logit.append(outputs.data.cpu())
                val_loss = np.mean(val_loss)
                _, _, prec_rec_f1, roc_auc, prc_auc = calc_all_scores(torch.cat(y_true), torch.cat(y_prob_logit),
                                                                      val_loss, "Valid", epoch)
                val_f1 = prec_rec_f1[-1:][0]
                # if val_loss < best_loss:
                # if val_f1 > best_f1:
                if prc_auc is None or np.isnan(prc_auc):
                    prc_auc = best_prc_auc
                if prc_auc > best_prc_auc:
                    best_loss = val_loss
                    best_f1 = val_f1
                    best_prc_auc = prc_auc
                    best_epoch = epoch
                    torch.save(net.state_dict(), config.out_checkpoint_path)
                    logging.info(f"Saving best model so far at epoch {epoch} with f1 = {val_f1} and au prc = {prc_auc}")

        logging.info("Finished Training")
        logging.info(f"Saved best model state dict to {config.out_checkpoint_path}")
        logging.info(f"Best model was achieved after {best_epoch} epochs, with val loss = {best_loss}")

    # TESTING

    logging.info("Beginning test...")
    model_path = args.in_checkpoint_path if args.only_get_metrics else config.out_checkpoint_path
    logging.info(f"Loading model from {model_path}")
    net.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        y_true = []
        y_prob_logit = []
        test_loss = []
        roi_types = []
        net.eval()
        logging.info("Testing...")
        for i, data in enumerate(tqdm(test_dataloader)):
            samples, labels, _, roi_type_arr = data
            # logging.info(images.size())
            outputs = net(samples.to(device))
            test_loss.append(criterion(outputs.cpu(), labels))
            y_true.append(labels)
            y_prob_logit.append(outputs.data.cpu())

            roi_types.extend(roi_type_arr)
        test_loss = np.mean(test_loss)
        y_true = torch.cat(y_true)
        y_prob_logit = torch.cat(y_prob_logit)
        calc_all_scores(y_true, y_prob_logit, test_loss, "Test")

        mass_indices = [i for i, item in enumerate(roi_types) if item == 'Mass' or item == 'healthy']
        calc_AUROC(y_true[mass_indices], torch.exp(y_prob_logit)[mass_indices], "Test only Masses")
        calc_AUPRC(y_true[mass_indices], torch.exp(y_prob_logit)[mass_indices], "Test only Masses")

        calcification_indices = [i for i, item in enumerate(roi_types) if item == 'Calcification' or item == 'healthy']
        calc_AUROC(y_true[calcification_indices], torch.exp(y_prob_logit)[calcification_indices],
                   "Test only Calcifications")
        calc_AUPRC(y_true[calcification_indices], torch.exp(y_prob_logit)[calcification_indices],
                   "Test only Calcifications")

        if config.classify_binary_healthy: output_ROC_curve(y_true, y_prob_logit, "Test", logfilename)
    logging.info("Finished testing.")
