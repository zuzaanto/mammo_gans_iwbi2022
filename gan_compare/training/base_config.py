import logging
from dataclasses import dataclass, field
from time import time
from typing import List


@dataclass
class BaseConfig:

    # model type and task (CLF / GAN) not known. Overwritten in specific configs
    model_name: str = "unknown"

    # Overwritten in specific configs
    output_model_dir: str = f"model_checkpoints/training_{model_name}_{time()}/"

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
    # OR for classification, the number of classes (set automatically though in classification_config.py)
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

    ngpu: int = 1

    dataset_names: List[str] = field(default_factory=list)

    # Whether to train conditional GAN
    conditional: bool = False

    # We can condition on different variables such as breast density or birads status of lesion. Default = "density"
    conditioned_on: str = "density"

    # Specifiy whether birads condition is modeled as binary e.g., benign/malignant with birads 1-3 = 0, 4-6 = 1
    is_condition_binary: bool = False

    # Preprocessing of training images
    # Variables for utils.py -> get_measures_for_crop():
    zoom_offset: float = 0.2  # the higher, the more likely the patch is zoomed out. if 0, no offset. negative means, patch is rather zoomed in
    zoom_spread: float = (
        0.33  # the higher, the more variance in zooming. must be greater 0.
    )
    ratio_spread: float = 0.05  # NOT IN USE ANYMORE. coefficient for how much to spread the ratio between height and width. the higher, the more spread.
    translation_spread: float = (
        0.25  # the higher, the more variance in translation. must be greater 0.
    )
    max_translation_offset: float = 0.33  # coefficient relative to the image size.

    def __post_init__(self):
        if self.model_name == "swin_transformer":
            self.image_size = (
                224  # swin transformer currently only supports 224x224 images
            )
            self.nc = 3
            logging.info(
                f"Changed image shape to {self.image_size}x{self.image_size}x{self.nc}, as is needed for the selected model ({self.model_name})"
            )
