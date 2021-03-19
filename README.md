# Gans for mammography lesions generation
This topic may change with time..
## Usage
Before running any code, remember to:
- install dependencies from `requirements.txt` and
- specify correct data paths in `gan_compare/paths.py`.

#### Generating metadata
To be able to read the dataset class at training time, you first need to generate a metadata file with all bboxes you will use in training. To do that, run:
```
python -m gan_compare.scripts.create_metadata --output_path where/to/save/metadata.json

```

#### Training

To start your DCGAN training, run:
```
python -m gan_compare.training.train \
--model_name MODEL_NAME \ #Model name: supported: dcgan and lsgan
--save_dataset SAVE_DATASET \ #Boolean indicating whether to generate and save a synthetic dataset
--out_dataset_path OUT_DATASET_PATH \ #Directory to save the dataset samples in.
--in_metadata_path IN_METADATA_PATH \ #Path to metadata.json file.

```
Note that you can change any parameters you want via `gan_compare/training/config.py`

For LSGAN, only 64x64 is supported.
For DCGAN, currently supported are 2 image sizes: 64x64 and 128x128.

Note that you can use LS loss with DCGAN - for more details check the `gan_compare/training/config.py`. Actually, 128x128 images will cause vanishing gradient in DCGAN, unless you use LS loss.

#### Inference

To generate images with your pretrained GAN, run:
```
python gan_compare.training.generate \
--model_name MODEL_NAME \ #Model name: supported: dcgan and lsgan
--image_size IMAGE_SIZE \ #Image size: 64 or 128
--model_checkpoint_path MODEL_CHECKPOINT_PATH \ #Path to model checkpoint
--num_samples NUM_SAMPLES \ #How many samples to generate
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

## Future work

1. [Interesting page](https://github.com/soumith/ganhacks) with a casual summary of GAN practical training knowledge, tips and tricks.
2. An interesting [paper](https://arxiv.org/pdf/1606.03498.pdf) about improvements in training GANs, helping to - among all else - tackle mode collapse issue. (It's from 2016, so quite old, but seems still respected in the community, and wwas written by.. well, legendary people)
