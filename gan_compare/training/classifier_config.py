from dataclasses import dataclass, field
from typing import List
from gan_compare.constants import DATASET_DICT


@dataclass
class ClassifierConfig:
    model_name = "cnn"

    # Paths to train and validation metadata
    train_metadata_path: str
    validation_metadata_path: str
    test_metadata_path: str
    
    # Path to synthetic metadata used for data augmentation
    synthetic_metadata_path: str
    train_shuffle_proportion: float = 0.4
    validation_shuffle_proportion: float = 0

    # Directory with synthetic patches
    synthetic_data_dir: str = None
    
    # Proportion of training artificial images
    gan_images_ratio: float = 0.4

    # Birads range
    birads_min: int = 2
    birads_max: int = 6

    split_birads_fours: bool = True

    # Whether to do binary classification of healthy/non-healthy patches
    classify_binary_healthy: bool = False

    # Whether to use synthetic data at all
    use_synthetic: bool = True

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

    # Dropout rate
    dropout_rate: float = 0.3

    # Number of training epochs
    num_epochs: int = 60

    # Learning rate for optimizers
    lr: float = 0.0002

    ngpu: int = 1

    out_checkpoint_path: str = "model_checkpoints//classifier/classifier.pt"

    dataset_names: List[str] = field(default_factory=list)
    
    classes: str = "is_healthy"

    conditional: bool = False

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
