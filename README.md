# Gans for mammography lesions generation

![Build](https://github.com/zuzaanto/mammo_gans/actions/workflows/python-app.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This topic may change with time..

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

## Usage
Before running any code, remember to:
- install dependencies from `requirements.txt` using pip or from `Pipfile` using pipenv and
- specify correct data paths in `gan_compare/paths.py`.

#### Generating metadata
To be able to read the dataset class at training time, you first need to generate a metadata file with all bboxes you will use in training. To do that, run:
```
python -m gan_compare.scripts.create_metadata --output_path where/to/save/metadata.json

```

#### Training

To start your DCGAN training, run:
```
python -m gan_compare.training.train_gan \
--model_name MODEL_NAME \ # Model name: supported: dcgan and lsgan
--config_path CONFIG_PATH \ # Path to a yaml model config file
--save_dataset SAVE_DATASET \ # Indicating whether to generate and save a synthetic dataset
--out_dataset_path OUT_DATASET_PATH \ # Directory to save the dataset samples in.
--in_metadata_path IN_METADATA_PATH \ # Path to metadata json file.
```

Note that you can change any parameters you want by passing the correct config yaml file. A sample and default config file can be found in `gan_compare/configs/dcgan_config.py`.

For LSGAN, only 64x64 is supported.
For DCGAN, currently supported are 2 image sizes: 64x64 and 128x128.

Note that you can use LS loss with DCGAN - for more details check the `gan_compare/training/gan_config.py`. Actually, 128x128 images may cause vanishing gradient in DCGAN, unless you use LS loss.

Note that you can condition the DCGAN training on BIRADS. BIRADS labels determine risk of malignancy and are assigned to each region of interest in the mammogram.

#### Inference

To generate images with your pretrained GAN, run:
```
python -m gan_compare.training.generate \
--model_name MODEL_NAME \ #Model name: supported: dcgan and lsgan
--model_checkpoint_dir MODEL_CHECKPOINT_DIR \ # Path to model checkpoint directory
--model_checkpoint_path MODEL_CHECKPOINT_PATH \ # Path to model checkpoint .pt file (optional, by default takes model.pt file in model_checkpoint_dir)
--num_samples NUM_SAMPLES \ #Number of samples to be generated
--birads BIRADS \ #The BIRADS risk status (int, 1-6). Control sample generation using a cGAN conditioned on BIRADS.
```
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

#### Train classifier
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
--output_dir OUTPUT_DIR # Directory to save 3 new metadata files.
```
After that, you can train and evaluate the classifier as follows:
```
python -m gan_compare.scripts.train_test_classifier \
  --config_path CONFIG_PATH # Path to a yaml model config file
  --only_get_metrics # Whether to skip training and only output the metrics on the test set. If true, requires in_checkpoint_path
  --in_checkpoint_path IN_CHECKPOINT_PATH # Path to a model checkpoint file, which will be used for outputting the metrics on the test set
  --save_dataset # Only use this if you just want to output your dataset as images, but don't want to perform any training our classifying
```

#### Peek through the dataset
This script walks through your data directory and shows you InBreast images with overlayed ground-truth masks:
```
python -m gan_compare.data_utils.data_utils.read_inbreast_image

```
The following script makes it easier to peek the statistics of a particular metadata file:
```
python -m gan_compare.scripts.get_metadata_statistics --metadata_path path/to/metadata.json
```
There is an additional folder for Jupyter notebooks oriented around better dataset understanding - `gan_compare/data_utils/eda/`

#### Visualize training on tensorboard
Tensorboard is integrated to facilitate visual analysis of GAN losses, discriminator accuracy and model architecture. 
During training, tensorboard writes some event logs that will later be used to generate the tensorboard visualizations. 
By default, these logs are stored in `gan_compare/model_checkpoint/training_{TIMESTAMP}/visualization`

To run tensorboard on localhost, adjust and run the following command:
```
tensorboard --logdir=gan_compare/model_checkpoint/training_{TIMESTAMP}/visualization
```
Note: if TensorBoard cannot find your event files, try using an absolute path for logdir.

## Future work

1. [Interesting page](https://github.com/soumith/ganhacks) with a casual summary of GAN practical training knowledge, tips and tricks.
2. An interesting [paper](https://arxiv.org/pdf/1606.03498.pdf) about improvements in training GANs, helping to - among all else - tackle mode collapse issue. (It's from 2016, so quite old, but seems still respected in the community, and was written by.. well, legendary people)
