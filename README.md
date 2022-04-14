# GANs for mammography lesions generation

![Build](https://github.com/zuzaanto/mammo_gans/actions/workflows/python-app.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation
To install this repo, first create your virtual environment in Python3.8, for example this way:

```
python3.8 -m venv my_venv
```
And then install the requirements as follows:
```
pip install -r requirements.txt
```
Or, if you use pipenv, you can create an environment and install requirements from Pipfile using:
```
pipenv shell
pipenv install
```

## Setup
Before running any code, remember to:
- install dependencies from `requirements.txt` using pip or from `Pipfile` using pipenv and
- specify correct data paths in `gan_compare/paths.py`.

## Generating metadata
To be able to read the dataset class at training time, you first need to generate a metadata file with all bboxes you will use in training. To do that, run:
```
python -m gan_compare.scripts.create_metadata --output_path where/to/save/metadata.json

```

### Train GANs

To start GAN training, run:
```
python -m gan_compare.training.train_gan \
--config_path CONFIG_PATH \ # Path to a yaml model config file
--save_dataset SAVE_DATASET \ # Indicating whether to save the gan input dataset constructed from the metadata.
--out_dataset_path OUT_DATASET_PATH \ # Directory to save the dataset samples in.
--seed SEED \ # The random seed for experiment reproducibility. 
```

Note that you can change any parameters you want by passing the correct config yaml file. For example, the type of GAN and the metadata (e.g. training dataset) are specificied in the config.yaml. 
Sample default config files can be found in `gan_compare/configs/gan/`.

##### GAN Types

 GAN Type       | WGAN-GP      |      DCGAN    |   LSGAN
-------------   | ------------- | ------------- | -------------
default config in `gan_compare/configs/gan/` | `wgangp_config.yaml`  | `dcgan_config.yaml`  | `lsgan_config.yaml` 
supported image sizes  | 224x224, 128x128, 64x64  | 224x224, 128x128, 64x64  | 64x64

Note that you can use LS loss with DCGAN. For 224x224 WGAN-GP achieves the best visual results.

Note that you can condition the GAN training (CGAN). You can condition on `'BIRADS'` or `'Density'`. BIRADS labels (1-6) determine the risk of malignancy of a lesion and are assigned to each region of interest in the mammogram. The ACR BIRADS breast density (1-4) is assigned on mammogram-level based on the amount of fibroglandular tissue of the breast. 

Note, that if the config.yaml of the model you use to generate samples was trained with the variable `split_birads_fours=True`, then the BIRADS conditions have been translated as follows: 
`{
    "2": 1,
    "3": 2,
    "4a": 3,
    "4b": 4,
    "4c": 5,
    "5": 6,
    "6": 7
}`

#### Visualize training on tensorboard
Tensorboard is integrated to facilitate visual analysis of GAN losses, discriminator accuracy and model architecture. 
During training, tensorboard writes some event logs that will later be used to generate the tensorboard visualizations. 
By default, these logs are stored in `gan_compare/model_checkpoint/training_{TIMESTAMP}/visualization`

To run tensorboard on localhost, adjust and run the following command:
```
tensorboard --logdir=gan_compare/model_checkpoint/training_{TIMESTAMP}/visualization
```
Note: if TensorBoard cannot find your event files, try using an absolute path for logdir.

#### Inference

To generate images with your pretrained GAN, run:
```
python -m gan_compare.training.generate \
--model_checkpoint_dir MODEL_CHECKPOINT_DIR \ # Path to model checkpoint directory, in which the gan config yaml file will be searched for.
--model_checkpoint_path MODEL_CHECKPOINT_PATH \ # Path to model checkpoint .pt file (optional, by default takes model.pt file in model_checkpoint_dir)
--dont_show_images DONT_SHOW_IMAGES \ # Indicating whether an image viewer should be used to displaya the generated images.
--save_images SAVE_IMAGES \ # Indicating whether to save the synthetic images
--out_images_path OUT_IMAGES_PATH \ # The location on the file system where the generated samples should be stored
--num_samples NUM_SAMPLES \ #Number of samples to be generated
--condition CONDITION \ #e.g. the BIRADS risk status (int, 1-6) or or breast density (1-4). Control sample generation using a cGAN conditioned on a continuous or discrete label.
--device DEVICE \ #The device on which inference should be run, e.g. "cuda" for GPU or "cpu" for CPU 
--seed SEED \ # The random seed for experiment reproducibility. 
```


### Train classifier
To train a classifier using partially synthetic data, you first need to generate synthetic images and metadata:
```
python -m gan_compare.scripts.create_metadata_from_checkpoint \
  --output_path OUTPUT_PATH \ # Path to json file to store metadata in.
  --checkpoint_path CHECKPOINT_PATH \ # Path to model's checkpoint.
  --generated_data_dir GENERATED_DATA_DIR \ # Directory to save generated images in.
  --num_samples_per_class NUM_SAMPLES_PER_CLASS \ # Number of samples to generate per each class.
  --model_config_path MODEL_CONFIG_PATH \ # Path to model config file.
  --model_name MODEL_NAME # Model name.
```
You also need to split your original, real metadata into train, validation and test subsets:
```
python -m gan_compare.scripts.split_metadata \
--metadata_path METADATA_PATH \ # Path to json file with metadata.
[--train_proportion TRAIN_PROPORTION] \ # Proportion of train subset.
[--val_proportion VAL_PROPORTION] \ # Proportion of val subset.
--output_path OUTPUT_PATH # Path to save the json split file in.
```
After that, you can train and evaluate the classifier as follows:
```
python -m gan_compare.scripts.train_test_classifier \
  --config_path CONFIG_PATH # Path to a yaml model config file
  --only_get_metrics # Whether to skip training and only output the metrics on the test set. If true, requires in_checkpoint_path
  --in_checkpoint_path IN_CHECKPOINT_PATH # Path to a model checkpoint file, which will be used for outputting the metrics on the test set
  --save_dataset # Only use this if you just want to output your dataset as images, but don't want to perform any training our classifying
```

### Peek through the dataset
This script walks through your data directory and shows you InBreast images with overlayed ground-truth masks:
```
python -m gan_compare.data_utils.data_utils.read_inbreast_image

```
The following script makes it easier to peek the statistics of a particular metadata file:
```
python -m gan_compare.scripts.get_metadata_statistics --metadata_path path/to/metadata.json
```
There is an additional folder for Jupyter notebooks oriented around better dataset understanding - `gan_compare/data_utils/eda/`

## Citations
Please consider citing the following articles:
1. Szafranowska, Zuzanna, et al. "Sharing Generative Models Instead of Private Data: A Simulation Study on Mammography Patch Classification." arXiv preprint arXiv:2203.04961 (2022). [Link](https://arxiv.org/abs/2203.04961)
2. Osuala, Richard, et al. "A review of generative adversarial networks in cancer imaging: New applications, new solutions." arXiv preprint arXiv:2107.09543 (2021). [Link](https://arxiv.org/abs/2203.04961)
