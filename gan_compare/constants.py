import logging
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from gan_compare.training.networks.classification.classifier_64 import Net as Net64
from gan_compare.training.networks.classification.classifier_128 import Net as Net128
from gan_compare.training.networks.classification.swin_transformer import (
    SwinTransformer,
)

DATASET_LIST = ["bcdr", "inbreast", "cbis-ddsm"]


def get_classifier(config: dataclass, num_classes: Optional[int] = None) -> nn.Module:

    if num_classes is None:
        # FIXME During GAN training config.n_cond is the number of conditions and not the number of classes.
        # Workaround: Usage of value from num_classes attribute instead.
        num_classes = config.num_classes
    if config.model_name == "swin_transformer":
        if not config.image_size == 224:
            logging.warning(
                f"For {config.model_name}, you are using image_size {config.image_size}, while the default image shape is 224x224x3."
            )
        return SwinTransformer(num_classes=num_classes, img_size=config.image_size)
    if config.model_name == "cnn":
        return_probabilities = (
            False
            if hasattr(config, "pretrain_classifier") is False
            else config.pretrain_classifier
        )
        if config.image_size == 64:
            return Net64(
                num_labels=num_classes, return_probabilities=return_probabilities
            )
        elif config.image_size == 128:
            return Net128(
                num_labels=num_classes, return_probabilities=return_probabilities
            )
        raise ValueError(f"Unrecognized CNN image size = {config.image_size}")
    raise ValueError(f"Unrecognized model name = {config.model_name}")
