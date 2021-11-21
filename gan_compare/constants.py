from gan_compare.dataset.inbreast_dataset import InbreastDataset
from gan_compare.dataset.bcdr_dataset import BCDRDataset
from gan_compare.training.networks.classification.swin_transformer import SwinTransformer
from gan_compare.training.networks.classification.classifier_64 import Net as Net64
from gan_compare.training.networks.classification.classifier_128 import Net as Net128
import torch.nn as nn


DATASET_DICT = {
    "bcdr": BCDRDataset,
    "inbreast": InbreastDataset,
}

DENSITIES = [1,2,3,4]

def get_classifier(name: str, img_size: int, num_classes: int) -> nn.Module:
    if name == "swin_transformer":
        return SwinTransformer(num_classes=num_classes, img_size=img_size)
    if name == "cnn":
        if img_size == 64:
            return Net64(num_labels=num_classes)
        elif img_size == 128:
            return Net128(num_labels=num_classes)
        raise ValueError(f"Unrecognized CNN image size = {img_size}")
    raise ValueError(f"Unrecognized model name = {name}")
