from __future__ import print_function

import logging
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

try:
    import tkinter
except:
    # Need to use matplotlib without tkinter dependency
    # tkinter is n.a. in some python distributions
    import matplotlib

    matplotlib.use("Agg")
from typing import Optional, Union

from gan_compare.constants import get_classifier
from gan_compare.dataset.constants import DENSITY_DICT
from gan_compare.training.gan_config import GANConfig
from gan_compare.training.io import save_yaml
from gan_compare.training.networks.generation.utils import (
    weights_init,
)
from gan_compare.training.visualization import VisualizationUtils


class BaseGANModel:
    def __init__(
            self,
            config: GANConfig,
            dataloader: DataLoader,
    ):
        self.config = config
        self.dataloader = dataloader

        # Asserts and checks
        self._assert_network_channels()

        # Decide which device we want to run on
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.config.ngpu > 0) else "cpu"
        )

        # Set further configuration params
        self.output_model_dir = Path(self.config.output_model_dir)
        self.are_outputs_logits = None  # Outputs of D can be probabilities or logits. We need to handle both cases.

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        # For convenience, let's use the specified batch size = number of fixed noise random tensors
        self.fixed_noise = torch.randn(
            self.config.batch_size, self.config.nz, 1, 1, device=self.device
        )

        # Create batch of fixed conditions that we will use to visualize the progression of the generator
        self.fixed_condition = None if self.config.conditional else self._get_random_conditions()

        # visualize model in tensorboard and instantiate visualizationUtils class object
        self.visualization_utils = self.visualize(
            fixed_noise=self.fixed_noise, fixed_condition=self.fixed_condition
        )

        # Handle init of classifier D2 pretraining
        if self.config.pretrain_classifier:
            if self.config.is_pretraining_adversarial:  # real/fake prediction
                num_classes = 1
            self.netD2 = get_classifier(self.config, num_classes=num_classes).to(self.device)
            self.netD2 = self._network_weights_init(net=self.netD2)
            self.netD2 = self._handle_multigpu(net=self.netD2)
            self._print_network_info(net=self.netD2)
            self.optimizerD2 = self.optimizer_setup(
                net=self.netD2,
                lr=self.config.lr_d2,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )

        # As we train a new model (atm no continued checkpoint training), we create new model dir and save config.
        self._save_config()


    def _assert_network_channels(self):
        """ Check if channel size is correct """

        if self.config.conditional:
            assert self.config.nc == 2 or (
                    self.config.nc == 4 and self.config.model_name == "swin_transformer"
            ), "To use conditional input, change number of channels (nc) to 2 (default) or 4 (swin transformer)."
        else:
            assert self.config.nc == 1 or (
                    self.config.nc == 3 and self.config.model_name == "swin_transformer"
            ), "Without conditional input into GAN, change number of channels (nc) to 1 (default) or 3 (swin transformer)."

    def network_setup(self, netD, netG) -> (nn.Module, nn.Module):
        """ Wrapper function to init weights and multigpu, and print network architecture """

        netD = self._network_weights_init(net=netD)
        netD = self._handle_multigpu(net=netD)
        self._print_network_info(net=netD)

        netG = self._network_weights_init(net=netG)
        netG = self._handle_multigpu(net=netG)
        self._print_network_info(net=netG)
        return netD, netG

    def create_optimizer(self, net, lr, betas, type="Adam", weight_decay=0):
        # Setup Adam optimizers for both G and D
        assert type != "Adam", "Currently only optim.Adam is implemented. Please extend code if you want to use other optimizers "
        return optim.Adam(
            net.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )


    def optimizer_setup(self, netD, netG):
        """ Setup Adam optimizers for both G and D """

        # Setup Adam optimizers for both G and D
        optimizerD = self.create_optimizer(
            net=self.netD,
            lr=self.config.lr_d1,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        optimizerG = self.create_optimizer(
            net=self.netG,
            lr=self.config.lr_g,
            betas=(self.config.beta1, self.config.beta2),
        )
        return optimizerD, optimizerG

    def _network_weights_init(self, net) -> nn.Module:
        """ Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2. """

        return net.apply(weights_init)

    def _handle_multigpu(self, net) -> nn.Module:
        """ Handle multi-gpu if desired """

        if (self.device.type == "cuda") and (self.config.ngpu > 1):
            return nn.DataParallel(net, list(range(self.config.ngpu)))
        return net

    def _print_network_info(self, net):
        """Print the network architecture """

        logging.info(net)

    def _mkdir_model_dir(self):
        """ Create folder where GAN will be stored """

        if not self.output_model_dir.exists():
            os.makedirs(self.output_model_dir.resolve())

    def _save_config(self, config_file_name: str = f"config.yaml"):
        """ Save the config to disc """

        self._mkdir_model_dir()  # validation to make sure model dir exists
        out_config_path = self.output_model_dir / config_file_name
        save_yaml(path=out_config_path, data=self.config)
        logging.info(f"Saved model config to {out_config_path.resolve()}")

    def _save_model(self, epoch_number: Optional[int] = None):
        """ Save the model to disc """

        self._mkdir_model_dir()  # validation to make sure model dir exists
        if epoch_number is None:
            out_path = self.output_model_dir / "model.pt"
        else:
            out_path = self.output_model_dir / f"{epoch_number}.pt"
        d = {
            "discriminator": self.netD.state_dict(),
            "generator": self.netG.state_dict(),
            "optim_discriminator": self.optimizerD.state_dict(),
            "optim_generator": self.optimizerG.state_dict(),
        }
        if self.config.pretrain_classifier:
            d["discriminator2"] = self.netD2.state_dict()
            d["optim_discriminator2"] = self.optimizerD2.state_dict()
        # Saving the model in out_path
        torch.save(d, out_path)
        logging.info(
            f"Saved model (on epoch(?): {epoch_number}) to {out_path.resolve()}"
        )
        # emptying the cache to avoid "CUDA out of memory"
        torch.cuda.empty_cache()

    def init_running_losses(self, init_value=0.0):
        return init_value, init_value, init_value, init_value, [], []

    def _netD_backward_pass(
            self,
            output,
            label_as_float: float,
            epoch: int,
            are_outputs_logits: bool = False,
    ):
        """ Backward pass through discriminator network"""

        raise NotImplementedError

    def _netG_update(
            self,
            netD,
            fake_images,
            fake_conditions,
            epoch: int,
            are_outputs_logits: bool = False,
            retain_graph: bool = False,
            is_G_updated: bool = True,
    ):
        """Update Generator network: e.g. in dcgan the goal is to maximize log(D(G(z)))"""

        raise NotImplementedError


    def _compute_loss(
        self,
        output,
        label,
        epoch: int,
        get_loss_for: str,
        are_outputs_logits: bool = False,
        netD=None,
        real_images=None,
        fake_images=None,
    ):
        """Setting the loss function. Computing and returning the loss."""

        raise NotImplementedError


    def _get_labels(self, smoothing: bool = True):
        """ get the labels and opionally smooth the real label """

        if self.config.use_one_sided_label_smoothing and smoothing:
            # if enabled, let's smooth the labels for "real" (--> real !=1)
            smoothed_real_label: float = random.uniform(
                self.config.label_smoothing_start, self.config.label_smoothing_end
            )
            logging.debug(f"smoothed_real_label = {smoothed_real_label}")
            return {"real": smoothed_real_label, "fake": 0.0}
        return {"real": 1.0, "fake": 0.0}

    def _get_random_conditions(
            self, minimum=None, maximum=None, batch_size=None, requires_grad=False
    ):
        """ Get randomised conditions between min and max for cGAN input """

        if minimum is None:
            minimum = self.config.condition_min

        if maximum is None:
            # Need to add +1 here to allow torch.rand/randint to create samples with number = condition_max
            maximum = self.config.condition_max + 1

        if batch_size is None:
            # sometimes we might want to pass batch_size i.e. in the last batch of an epoch that might have less
            # training samples than previous batches.
            batch_size = self.config.batch_size

        if (
                self.config.conditioned_on == "density"
                and not self.config.is_condition_categorical
                and not self.config.is_condition_binary
        ):
            # here we need a float randomly drawn from a set of possible values (0.0, 0.33, 0.67, 1.0) for breast density (1 - 4)
            conditions = []
            condition_value_options = list(DENSITY_DICT.values())
            for i in range(batch_size):
                # number out of [-1,1] multiplied by noise term parameter. Round for 2 digits
                noise = round(random.uniform(-1, 1) * self.config.added_noise_term, 2)
                # get condition with noise normalised between 0 and 1.
                condition_w_noise = max(
                    min(random.choice(condition_value_options) + noise, 1.0), 0.0
                )
                conditions.append(condition_w_noise)
            condition_tensor = torch.tensor(
                conditions, device=self.device, requires_grad=requires_grad
            )
            logging.debug(f"random condition_tensor: {condition_tensor}")
            return condition_tensor
        else:
            # now we want an integer rather than a float.
            return torch.randint(
                minimum,
                maximum,
                (batch_size,),
                device=self.device,
                requires_grad=requires_grad,
            )


    def handle_G_updates(
            self,
            iteration: int,
            fake_images,
            fake_conditions,
            epoch: int,
            b_size: int = None,
            is_D2_using_new_fakes: bool = True,
    ):
        """ Generator updates based on one or multiple (non-)backpropagating discriminators """

        # return variable init
        output_fake_2_D2 = None
        D2_G_z = None
        errG_2 = None

        if self.config.pretrain_classifier:
            # In case D2 only backpropagates after a certain number of epochs has passed.
            is_D2_backpropagated: bool = (
                    epoch >= self.config.start_backprop_D2_into_G_after_epoch
            )
            if (
                    epoch == self.config.start_backprop_D2_into_G_after_epoch
                    and iteration == 0
            ):
                logging.info(
                    f"As we have reached epoch={epoch}, we now start to backpropagate into G the gradients of D2 ({self.config.model_name})."
                )
            # Swin transformer returns last layer logits instead of probabilities
            are_outputs_logits = (
                True if self.config.model_name == "swin_transformer" else False
            )

        # Checking which D should backpropagate into G in this epoch.
        if (
                self.config.pretrain_classifier
                and self.config.are_Ds_alternating_to_update_G
                and is_D2_backpropagated
        ):

            # We always pass the two outputs of D1 and D2 through G, but only update with one of the outputs.
            if iteration % 2 == 0:
                # D1 output passed through G, AND backpropagated
                output_fake_2_D1, D_G_z2, errG = self._netG_update(
                    netD=self.netD,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=False,
                    is_G_updated=True,
                )
                # D2 output passed through G, NOT backpropagated
                output_fake_2_D2, D2_G_z, errG_2 = self._netG_update(
                    netD=self.netD2,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=are_outputs_logits,
                    is_G_updated=False,
                )
            else:
                # D1 output passed through G, NOT backpropagated
                output_fake_2_D1, D_G_z2, errG = self._netG_update(
                    netD=self.netD,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=False,
                    is_G_updated=False,
                )
                # D2 output passed through G, AND backpropagated
                output_fake_2_D2, D2_G_z, errG_2 = self._netG_update(
                    netD=self.netD2,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=are_outputs_logits,
                    is_G_updated=True,
                )
        else:

            output_fake_2_D1, D_G_z2, errG = self._netG_update(
                netD=self.netD,
                fake_images=fake_images,
                fake_conditions=fake_conditions,
                retain_graph=self.config.pretrain_classifier,
                # another call to backward() will happen if we pretrain the classifier
                epoch=epoch,
                are_outputs_logits=False,
                is_G_updated=True,
            )

            if self.config.pretrain_classifier:
                self.netG.zero_grad()
                if is_D2_using_new_fakes:
                    # Generating new fake images as previous ones had been already incorporated in D's previous update
                    fake_images, fake_conditions = self.generate_during_training(
                        b_size=b_size
                    )
                output_fake_2_D2, D2_G_z, errG_2 = self._netG_update(
                    netD=self.netD2,
                    fake_images=fake_images,
                    fake_conditions=fake_conditions,
                    retain_graph=False,
                    epoch=epoch,
                    are_outputs_logits=are_outputs_logits,
                    is_G_updated=is_D2_backpropagated,
                )
        return output_fake_2_D1, D_G_z2, errG, output_fake_2_D2, D2_G_z, errG_2


    def train(self):
        """ Training the GAN network iterating over the dataloader """

        raise NotImplementedError


    def generate_during_training(self, b_size, noise=None):
        """Generate batch of latent vectors (& conditions) as input into generator to generate fake images"""

        fake_conditions = None
        if noise is None:
            noise = torch.randn(b_size, self.config.nz, 1, 1, device=self.device)
        if self.config.conditional:
            fake_conditions = self._get_random_conditions(batch_size=b_size)
            fake_images = self.netG(noise, fake_conditions)
        else:
            # Generate fake image batch with G (without condition)
            fake_images = self.netG(noise)

        return fake_images, fake_conditions

    def generate(
            self,
            model_checkpoint_path: Path,
            fixed_noise=None,
            fixed_condition=None,
            num_samples: int = 10,
            device: str = "cpu",
    ) -> list:
        """ Generate samples given a pretrained generator weights checkpoint"""

        self.optimizerG = optim.Adam(
            self.netG.parameters(),
            lr=self.config.lr_g,
            betas=(self.config.beta1, self.config.beta2),
        )
        map_location = "cpu" if device == "cpu" else "cuda:0"
        self.netG.to(device)
        self.netG.cpu() if device == "cpu" else self.netG.cuda()

        checkpoint = torch.load(model_checkpoint_path, map_location=map_location)
        self.netG.load_state_dict(checkpoint["generator"])
        # self.optimizerG.load_state_dict(checkpoint["optim_generator"])
        self.netG.eval()

        img_list = []
        # for ind in tqdm(range(num_samples)):
        if fixed_noise is None:
            fixed_noise = torch.randn(num_samples, self.config.nz, 1, 1, device=device)
        if self.config.conditional:
            if fixed_condition is None:
                fixed_condition = self._get_random_conditions(batch_size=num_samples)
            elif isinstance(fixed_condition, int):
                fixed_condition = self._get_random_conditions(
                    minimum=fixed_condition,
                    maximum=fixed_condition + 1,
                    batch_size=num_samples,
                )
            fake = self.netG(fixed_noise, fixed_condition).detach().cpu().numpy()
        else:
            fake = self.netG(fixed_noise).detach().cpu().numpy()
        img_list.extend(fake)
        return img_list

    def visualize(self, fixed_noise=None, fixed_condition=None):
        """visualization init for tensorboard logging of sample batches, losses, D accuracy, and model architecture"""

        with torch.no_grad():
            # we need the number of training iterations per epoch (depending on size of batch and training dataset)
            num_iterations_per_epoch = len(self.dataloader)

            # Setup visualizaion utilities, which includes tensorboard I/O functions
            visualization_utils = VisualizationUtils(
                num_iterations_per_epoch=num_iterations_per_epoch,
                num_iterations_between_prints=self.config.num_iterations_between_prints,
                output_model_dir=self.output_model_dir,
            )
            if fixed_noise is None:
                fixed_noise = torch.randn(
                    self.config.batch_size,
                    self.config.nz,
                    1,
                    1,
                    requires_grad=False,
                    device=self.device,
                )

            if self.config.conditional and fixed_condition is None:
                fixed_condition = self._get_random_conditions(requires_grad=False)

            # Visualize the model architecture of the generator
            visualization_utils.generate_tensorboard_network_graph(
                neural_network=self.netG,
                network_input_1=fixed_noise,
                network_input_2=fixed_condition,
            )
            return visualization_utils
