from gan_compare.dataset.inbreast_dataset import InbreastDataset
from gan_compare.dataset.bcdr_dataset import BCDRDataset
from gan_compare.training.networks.classification.swin_transformer import SwinTransformer
from gan_compare.training.networks.classification.classifier_64 import Net as Net64
from gan_compare.training.networks.classification.classifier_128 import Net as Net128
import torch.nn as nn


DATASET_DICT = {
    "bcdr": BCDRDataset,
    "inbreast": InbreastDataset,
    "bcdr_only_train": BCDRDataset,
}

DENSITIES = [1,2,3,4]

def get_classifier(config) -> nn.Module:
    if config.model_name == "swin_transformer":
        return SwinTransformer(num_classes=config.n_cond, img_size=config.image_size)
    if config.model_name == "cnn":
        return_probabilities = False if hasattr(config, "pretrain_classifier") is False else config.pretrain_classifier
        if config.image_size == 64:
            return Net64(num_labels=config.n_cond, return_probabilities=return_probabilities)
        elif config.image_size == 128:
            return Net128(num_labels=config.n_cond, return_probabilities=return_probabilities)
        raise ValueError(f"Unrecognized CNN image size = {config.image_size}")
    raise ValueError(f"Unrecognized model name = {config.name}")
