from dataclasses import dataclass, field
from typing import List
from gan_compare.constants import DATASET_DICT
from gan_compare.training.base_config import BaseConfig
from time import time

@dataclass
class ClassifierConfig(BaseConfig):
    # Paths to train and validation metadata
    train_metadata_path: str = None
    validation_metadata_path: str = None
    test_metadata_path: str = None

    model_name: str = "cnn"
    
    # Path to synthetic metadata used for data augmentation
    # synthetic_metadata_path: str # TODO REFACTOR

    # Different shuffle and sampling proportions
    train_shuffle_proportion: float = 0.5
    validation_shuffle_proportion: float  = 0.5
    training_sampling_proportion: float = 1.0

    # Directory with synthetic patches
    synthetic_data_dir: str = None

    
    # Proportion of training artificial images
    gan_images_ratio: float = 0.4

    no_transforms: bool = False

    # Whether to use synthetic data at all
    use_synthetic: bool = True

    # Dropout rate
    dropout_rate: float = 0.3

    #out_checkpoint_path: str = "model_checkpoints//classifier/best_classifier.pt"
    # Using time() to avoid overwriting existing model_checkpoints
    out_checkpoint_path: str = f"model_checkpoints/CLF_training_{time()}/best_classifier.pt"
    
    classes: str = "is_healthy"

    # Variables for utils.py -> get_measures_for_crop():
    zoom_offset: float = 0.2 # the higher, the more likely the patch is zoomed out. if 0, no offset. negative means, patch is rather zoomed in
    zoom_spread: float = 0.33 # the higher, the more variance in zooming. must be greater 0.
    ratio_spread: float = 0.05 # NOT IN USE ANYMORE. coefficient for how much to spread the ratio between height and width. the higher, the more spread.
    translation_spread: float = 0.25 # the higher, the more variance in translation. must be greater 0.
    max_translation_offset: float = 0.33 # coefficient relative to the image size.

    def __post_init__(self):
        if self.classify_binary_healthy:
            self.n_cond = 2
        elif self.split_birads_fours:
            self.birads_min = 1
            self.birads_max = 7
            self.n_cond = self.birads_max + 1
        assert self.classes in ["is_healthy", "birads"],  "Classifier currently supports either healthy vs unhealthy, or birads classification" # TODO Add ACR classification
        assert 1 >= self.train_shuffle_proportion >= 0, "Train shuffle proportion must be from <0,1> range"
        assert 1 >= self.validation_shuffle_proportion >= 0, "Validation shuffle proportion must be from <0,1> range"
        assert all(dataset_name in DATASET_DICT.keys() for dataset_name in self.dataset_names)
