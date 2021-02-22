# Gans for mammography lesions generation
This topic may change with time..
## Usage
Before running any code, remember to:
- install dependencies from `requirements.txt` and
- specify correct data paths in `gan_compare/paths.py`.
#### Training

To start your DCGAN training, run:
```
python gan_compare/training/train.py
```
Note that you can change any parameters you want via `gan_compare/training/config.py`

For DCGAN, currently supported are 2 image sizes: 64x64 and 128x128.

#### Peek through the dataset
This script walks through your data directory and shows you InBreast images with overlayed ground-truth masks:
```
python gan_compare/data_utils/read_inbreast_image.py
```

## Future work

1. [Interesting page](https://github.com/soumith/ganhacks) with a casual summary of GAN practical training knowledge, tips and tricks.
2. An interesting [paper](https://arxiv.org/pdf/1606.03498.pdf) about improvements in training GANs, helping to - among all else - tackle mode collapse issue. (It's from 2016, so quite old, but seems still respected in the community, and wwas written by.. well, legendary people)
