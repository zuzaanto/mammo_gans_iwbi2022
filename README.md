# Gans for mammography lesions generation
This topic may change with time..
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
python gan_compare/training/train.py --model_name <dcgan or lsgan>
```
Note that you can change any parameters you want via `gan_compare/training/config.py`

For LSGAN, only 64x64 is supported.
For DCGAN, currently supported are 2 image sizes: 64x64 and 128x128.

Note that you can use LS loss with DCGAN - for more details check the `gan_compare/training/config.py`. Actually, 128x128 images will cause vanishing gradient in DCGAN, unless you use LS loss.

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
