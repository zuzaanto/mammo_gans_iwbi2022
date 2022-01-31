#!/bin/sh
python -m gan_compare.scripts.classify_birads --config_path "C:\Users\Ben_Ji\Documents\repos\mammo_gans\gan_compare\configs\classification_no_synth_config.yaml"
python -m gan_compare.scripts.classify_birads --config_path "C:\Users\Ben_Ji\Documents\repos\mammo_gans\gan_compare\configs\classification_with_synth_config.yaml"