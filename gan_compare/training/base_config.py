from dataclasses import dataclass, field
from typing import List
from gan_compare.constants import DATASET_DICT


@dataclass
class BaseConfig:
    model_name: str = "dcgan"

    # Birads range
    birads_min: int = 2
    birads_max: int = 6

    seed: int = 42
    
    # 4a 4b 4c of birads are splitted into integers
    split_birads_fours: bool = True

    # Whether to do binary classification of healthy/non-healthy patches
    classify_binary_healthy: bool = False

    # Specify whether ROIs of calcifications should be included into GAN training
    is_trained_on_calcifications: bool = False

    # The number of condition labels for input into conditional GAN (i.e. 7 for BI-RADS 0 - 6)
    n_cond = birads_max + 1

    # Number of workers for dataloader
    workers: int = 2

    # Batch size during training
    batch_size: int = 8

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size: int = 64

    # Number of training epochs
    num_epochs: int = 60

    # Learning rate for optimizers
    lr: float = 0.0002

    ngpu: int = 1

    dataset_names: List[str] = field(default_factory=list)

    # Whether to train conditional GAN
    conditional: bool = False

    # We can condition on different variables such as breast density or birads status of lesion. Default = "density"
    conditioned_on: str = "density"

    # Specifiy whether birads condition is modeled as binary e.g., benign/malignant with birads 1-3 = 0, 4-6 = 1
    is_condition_binary: bool = False
