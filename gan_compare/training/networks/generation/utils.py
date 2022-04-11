import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.parallel
from torch import autograd, nn

from gan_compare.training.io import save_yaml


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def empty_cache():
    # Free up any used memory on each batch iteration
    import gc

    gc.collect()
    # emptying the cache to avoid "CUDA out of memory"
    torch.cuda.empty_cache()
    # Checking the memory consumption
    logging.debug(torch.cuda.memory_summary(device=None, abbreviated=False))


def init_running_losses(init_value=0.0):
    return init_value, init_value, init_value, init_value, [], []


def mkdir_model_dir(output_model_dir):
    """Create folder where GAN will be stored"""

    if not Path(output_model_dir).exists():
        os.makedirs(Path(output_model_dir).resolve())


def save_config(config, output_model_dir, config_file_name: str = f"config.yaml"):
    """Save the config to disc"""
    output_model_dir = Path(output_model_dir)
    mkdir_model_dir(
        output_model_dir=output_model_dir
    )  # validation to make sure model dir exists
    out_config_path = output_model_dir / config_file_name
    save_yaml(path=out_config_path, data=config)
    logging.info(f"Saved model config to {out_config_path.resolve()}")


def save_model(
    netD,
    optimizerD,
    netG,
    optimizerG,
    netD2,
    optimizerD2,
    output_model_dir,
    epoch_number: Optional[int] = None,
):
    """Save the model to disc"""
    output_model_dir = Path(output_model_dir)

    mkdir_model_dir(
        output_model_dir=output_model_dir
    )  # validation to make sure model dir exists
    if epoch_number is None:
        out_path = output_model_dir / "model.pt"
    else:
        out_path = output_model_dir / f"{epoch_number}.pt"
    d = {
        "discriminator": netD.state_dict(),
        "generator": netG.state_dict(),
        "optim_discriminator": optimizerD.state_dict(),
        "optim_generator": optimizerG.state_dict(),
    }
    if None not in (netD2, optimizerD2):
        d["discriminator2"] = netD2.state_dict()
        d["optim_discriminator2"] = optimizerD2.state_dict()
    # Saving the model in out_path
    torch.save(d, out_path)
    logging.info(f"Saved model (on epoch(?): {epoch_number}) to {out_path.resolve()}")
