# Gans for mammography lesions generation
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

## Usage
Before running any code, remember to:
- install dependencies from `requirements.txt` and
- specify correct data paths in `gan_compare/paths.py`.

#### Generating metadata
To be able to read the dataset class at training time, you first need to generate a metadata file with all bboxes you will use in training. To do that, run:
```
python gan_compare/scripts/create_metadata.py --output_path where/to/save/metadata.json
```

#### Training

To start your DCGAN training, run:
```
python gan_compare/training/train_gan.py \
--model_name <dcgan or lsgan> \
--config_path CONFIG_PATH \ # Path to a yaml model config file
--save_dataset \ # Whether to save the dataset samples.
--out_dataset_path OUT_DATASET_PATH # Directory to save the dataset samples in.
```
Note that you can change any parameters you want by passing the correct config yaml file. A sample and default config file can be found in `gan_compare/configs/dcgan_config.py`.

For LSGAN, only 64x64 is supported.
For DCGAN, currently supported are 2 image sizes: 64x64 and 128x128.

Note that you can use LS loss with DCGAN - for more details check the `gan_compare/training/gan_config.py`. Actually, 128x128 images will cause vanishing gradient in DCGAN, unless you use LS loss.

#### Inference

To generate images with your pretrained GAN, run:
```
python gan_compare/training/generate.py \
--model_name MODEL_NAME \ #Model name: supported: dcgan and lsgan
--model_checkpoint_dir MODEL_CHECKPOINT_DIR \ # Path to model checkpoint directory
--model_checkpoint_path MODEL_CHECKPOINT_PATH \ # Path to model checkpoint .pt file (optional, by default takes model.pt file in model_checkpoint_dir)
--num_samples NUM_SAMPLES \ #How many samples to generate
```

#### Peek through the dataset
This script walks through your data directory and shows you InBreast images with overlayed ground-truth masks:
```
python gan_compare/data_utils/read_inbreast_image.py
```
The following script makes it easier to peek the statistics of a particular metadata file:
```
python gan_compare/scripts/get_metadata_statistics.py --metadata_path path/to/metadata.json
```
There is an additional folder for Jupyter notebooks oriented around better dataset understanding - `gan_compare/data_utils/eda/`

## Future work

1. [Interesting page](https://github.com/soumith/ganhacks) with a casual summary of GAN practical training knowledge, tips and tricks.
2. An interesting [paper](https://arxiv.org/pdf/1606.03498.pdf) about improvements in training GANs, helping to - among all else - tackle mode collapse issue. (It's from 2016, so quite old, but seems still respected in the community, and wwas written by.. well, legendary people)
