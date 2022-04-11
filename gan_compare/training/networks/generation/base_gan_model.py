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
from gan_compare.training.networks.generation.utils import (
    init_running_losses,
    save_config,
    save_model,
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
        self.are_outputs_logits = None  # Outputs of D can be probabilities or logits. We need to handle both cases.

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        # For convenience, let's use the specified batch size = number of fixed noise random tensors
        self.fixed_noise = torch.randn(
            self.config.batch_size, self.config.nz, 1, 1, device=self.device
        )

        # Create batch of fixed conditions that we will use to visualize the progression of the generator
        self.fixed_condition = (
            self._get_random_conditions() if self.config.conditional else None
        )

        # Handle init of classifier D2 pretraining
        if self.config.pretrain_classifier:
            if self.config.is_pretraining_adversarial:  # real/fake prediction
                num_classes = 1
            self.netD2 = get_classifier(self.config, num_classes=num_classes).to(
                self.device
            )
            self.netD2 = self._network_weights_init(net=self.netD2)
            self.netD2 = self._handle_multigpu(net=self.netD2)
            self.optimizerD2 = self.create_optimizer(
                net=self.netD2,
                lr=self.config.lr_d2,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
            )
            logging.info(self.netD2)

        # As we train a new model (atm no continued checkpoint training), we create new model dir and save config.
        save_config(config=self.config, output_model_dir=self.config.output_model_dir)

    def _assert_network_channels(self):
        """Check if channel size is correct"""

        if self.config.conditional:
            assert self.config.nc == 2 or (
                self.config.nc == 4 and self.config.model_name == "swin_transformer"
            ), "To use conditional input, change number of channels (nc) to 2 (default) or 4 (swin transformer)."
        else:
            assert self.config.nc == 1 or (
                self.config.nc == 3 and self.config.model_name == "swin_transformer"
            ), "Without conditional input into GAN, change number of channels (nc) to 1 (default) or 3 (swin transformer)."

    def network_setup(
        self, netD, netG, is_visualized: bool = True
    ) -> (nn.Module, nn.Module):
        """Wrapper function to init weights and multigpu, print and optionally visualize network architecture"""

        netD = self._network_weights_init(net=netD)
        netD = self._handle_multigpu(net=netD)
        logging.info(netD)

        netG = self._network_weights_init(net=netG)
        netG = self._handle_multigpu(net=netG)
        logging.info(netG)

        if is_visualized:
            # visualize model in tensorboard and instantiate visualizationUtils class object
            self.visualization_utils = self.visualize(
                fixed_noise=self.fixed_noise, fixed_condition=self.fixed_condition
            )
        return netD, netG

    def create_optimizer(self, net, lr, betas, optim_type="Adam", weight_decay=0.0):
        """Create and return an optimizer"""

        assert (
            not optim_type != "Adam"
        ), "Currently only optim.Adam is implemented. Please extend code if you want to use other optimizers "
        return optim.Adam(
            params=net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

    def optimizer_setup(self, netD, netG):
        """Setup Adam optimizers for both G and D"""

        # Setup Adam optimizers for both G and D
        optimizerD = self.create_optimizer(
            net=netD,
            lr=self.config.lr_d1,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        optimizerG = self.create_optimizer(
            net=netG,
            lr=self.config.lr_g,
            betas=(self.config.beta1, self.config.beta2),
        )
        return optimizerD, optimizerG

    def _network_weights_init(self, net) -> nn.Module:
        """Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2."""

        return net.apply(weights_init)

    def _handle_multigpu(self, net) -> nn.Module:
        """Handle multi-gpu if desired"""

        if (self.device.type == "cuda") and (self.config.ngpu > 1):
            return nn.DataParallel(net, list(range(self.config.ngpu)))
        return net

    def _netD_forward_pass(self, netD, images, conditions=None):
        """Forward pass through discriminator network"""

        # Forward pass batch through D
        if self.config.conditional:
            output = netD(images, conditions).view(-1)
        else:
            output = netD(images).view(-1)

        # The mean prediction of the discriminator
        D_output_mean = output.mean().item()

        return output, D_output_mean

    def _netD_update(self, **kwargs):
        """Backward pass through discriminator network"""

        raise NotImplementedError

    def _netG_update(self, **kwargs):
        """Update Generator network: e.g. in dcgan the goal is to maximize log(D(G(z)))"""

        raise NotImplementedError

    def _compute_loss(self, **kwargs):
        """Setting the loss function. Computing and returning the loss."""

        raise NotImplementedError

    def _get_labels(
        self,
        b_size: int,
        label: str,
        smoothing: bool = True,
        real_label: str = 1.0,
        fake_label: str = 0.0,
    ):
        """get the labels as tensor and optionally smooth the real label values"""

        if label == "real":
            if smoothing:
                # if enabled, let's smooth the labels for "real" (--> real !=1)
                label_tensor = torch.FloatTensor(
                    size=(b_size,), device=self.device
                ).uniform_(
                    self.config.label_smoothing_start, self.config.label_smoothing_end
                )
            else:
                label_tensor = torch.full(
                    size=(b_size,), fill_value=real_label, device=self.device
                )
        elif label == "fake":
            label_tensor = torch.full(
                size=(b_size,), fill_value=fake_label, device=self.device
            )
            logging.debug(f"Created {label} labels tensor: {label_tensor}")
        else:
            raise Exception(
                f"label parameter needs to be defined as either 'real' or 'fake'. '{label}' was provided. Please adjust."
            )
        return label_tensor

    def _get_random_conditions(
        self, minimum=None, maximum=None, batch_size=None, requires_grad=False
    ):
        """Get randomised conditions between min and max for cGAN input"""

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
        """Generator updates based on one or multiple (non-)backpropagating discriminators"""

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
                    fake_images, fake_conditions = self._netG_forward_pass(
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

    def _netG_forward_pass(self, b_size, noise=None):
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
        """Generate samples given a pretrained generator weights checkpoint"""

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

    def is_D2_pretrained(self, epoch: int) -> bool:
        """Check if D2 to be included into training in current epoch and set pretrain_classifier accordingly."""

        # We check if netD2 was initialized, which means self.config.pretrain_classifier was true.
        if (
            hasattr(self, "netD2")
            and epoch >= self.config.start_training_D2_after_epoch
        ):
            if not self.config.pretrain_classifier or epoch == 0:
                logging.info(
                    f"As we have reached epoch={epoch}, we now start training D2 ({self.config.model_name})."
                )
            return True
        else:
            # We only want to train D2 after a certain number of epochs, hence we set self.config.pretrain_classifier = False
            # until that number of epochs is reached.
            return False

    def periodic_visualization_log(
        self,
        epoch,
        iteration,
        running_loss_of_generator,
        running_loss_of_discriminator,
        running_real_discriminator_accuracy,
        running_fake_discriminator_accuracy,
        running_loss_of_generator_D2=None,
        running_loss_of_discriminator2=None,
        running_real_discriminator2_accuracy=None,
        running_fake_discriminator2_accuracy=None,
    ):
        """wrapper adding multiple items of interest to visualization_utils"""

        # Add loss scalars to tensorboard
        self.visualization_utils.add_value_to_tensorboard_loss_diagram(
            epoch=epoch,
            iteration=iteration,
            running_loss_of_generator=running_loss_of_generator,
            running_loss_of_generator_D2=running_loss_of_generator_D2,
            running_loss_of_discriminator=running_loss_of_discriminator,
            running_loss_of_discriminator2=running_loss_of_discriminator2,
        )
        # Add accuracy scalars to tensorboard
        if None not in (
            running_real_discriminator_accuracy,
            running_fake_discriminator_accuracy,
        ):
            self.visualization_utils.add_value_to_tensorboard_accuracy_diagram(
                epoch=epoch,
                iteration=iteration,
                running_real_discriminator_accuracy=running_real_discriminator_accuracy,
                running_fake_discriminator_accuracy=running_fake_discriminator_accuracy,
                running_real_discriminator2_accuracy=running_real_discriminator2_accuracy,
                running_fake_discriminator2_accuracy=running_fake_discriminator2_accuracy,
            )

        # Visually check how the generator is doing by saving G's output on fixed_noise
        # if (iters % self.config.num_iterations_between_prints * 10 == 0) or (
        #        (epoch == self.config.num_epochs - 1) and (i == len(self.dataloader) - 1)):
        img_name: str = (
            "generated fixed-noise images each <"
            + str(self.config.num_iterations_between_prints)  # * 10)
            + ">th iteration. Epoch="
            + str(epoch)
            + ", iteration="
            + str(iteration)
        )
        self.visualization_utils.add_generated_batch_to_tensorboard(
            neural_network=self.netG,
            network_input_1=self.fixed_noise,
            network_input_2=self.fixed_condition,
            img_name=img_name,
        )

    def visualize(self, fixed_noise=None, fixed_condition=None):
        """visualization init for tensorboard logging of sample batches, losses, D accuracy, and model architecture"""

        with torch.no_grad():
            # we need the number of training iterations per epoch (depending on size of batch and training dataset)
            num_iterations_per_epoch = len(self.dataloader)

            # Setup visualizaion utilities, which includes tensorboard I/O functions
            visualization_utils = VisualizationUtils(
                num_iterations_per_epoch=num_iterations_per_epoch,
                num_iterations_between_prints=self.config.num_iterations_between_prints,
                output_model_dir=self.config.output_model_dir,
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

    def compute_discriminator_accuracy(
        self,
        output_real_D,
        running_real_D_acc,
        output_fake_D,
        running_fake_D_acc,
        output_real_D2=None,
        running_real_D2_acc=None,
        output_fake_D2=None,
        running_fake_D2_acc=None,
    ):
        """compute the current training accuracy metric"""

        # Calculate D's accuracy on the real data with real_label being = 1.
        # D1
        current_real_D_acc = (
            torch.sum(output_real_D > self.config.discriminator_clf_threshold).item()
            / list(output_real_D.size())[0]
        )

        # Calculate D's accuracy on the fake data from G with fake_label being = 0.
        # Note that we use the output_fake_D and not output_fake_2_D1, as 2 would be unfair,
        # as the discriminator has already received a weight update for the training batch
        current_fake_D_acc = (
            torch.sum(output_fake_D < self.config.discriminator_clf_threshold).item()
            / list(output_fake_D.size())[0]
        )

        running_real_D_acc += current_real_D_acc
        running_fake_D_acc += current_fake_D_acc

        # D2
        current_real_D2_acc = (
            None
            if output_real_D2 is None
            else (
                torch.sum(
                    output_real_D2 > self.config.discriminator_clf_threshold
                ).item()
                / list(output_real_D2.size())[0]
            )
        )
        current_fake_D2_acc = (
            None
            if output_fake_D2 is None
            else (
                torch.sum(
                    output_fake_D2 < self.config.discriminator_clf_threshold
                ).item()
                / list(output_fake_D2.size())[0]
            )
        )
        running_real_D2_acc = (
            running_real_D2_acc
            if current_real_D2_acc is None
            else running_real_D2_acc + current_real_D2_acc
        )
        running_fake_D2_acc = (
            running_fake_D2_acc
            if current_fake_D2_acc is None
            else running_fake_D2_acc + current_fake_D2_acc
        )

        return (
            running_real_D_acc,
            running_fake_D_acc,
            current_real_D_acc,
            current_fake_D_acc,
            running_real_D2_acc,
            running_fake_D2_acc,
            current_real_D2_acc,
            current_fake_D2_acc,
        )

    def train(self):
        """Training the GAN network iterating over the dataloader"""

        # initializing variables needed for visualization and # lists to keep track of progress
        (
            running_loss_of_generator,
            running_loss_of_discriminator,
            running_real_D_acc,
            running_fake_D_acc,
            G_losses,
            D_losses,
        ) = init_running_losses()

        iters = 0

        # set to None for function calls later below
        (
            running_loss_of_discriminator2,
            running_loss_of_generator_D2,
            running_real_D2_acc,
            running_fake_D2_acc,
            D2_losses,
            G2_losses,
        ) = init_running_losses(
            init_value=0.0 if self.config.pretrain_classifier else None
        )

        # Training Loop
        logging.info(
            f"Starting Training on {self.device}. Image size: {self.config.image_size}"
        ) if not self.config.conditional else logging.info(
            f"Starting Training on {self.device}. Image size: {self.config.image_size}. Training conditioned on: {self.config.conditioned_on}"
            f"{' as a continuous (with random noise: ' + f'{self.config.added_noise_term}' + ') and not' * (1 - self.config.is_condition_categorical)} as a categorical variable"
        )

        # For each epoch
        for epoch in range(self.config.num_epochs):

            # We check if we should include D2 into training in the current epoch by resetting config.pretrain_classifier
            self.config.pretrain_classifier = self.is_D2_pretrained(epoch=epoch)

            # d_iteration: For each n discriminator ("critic") updates we do 1 generator update.
            d_iteration: int = (
                self.config.d_iters_per_g_update
            )  # In the very first iteration 0, G is trained.

            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):

                # Unpack data (=image batch) alongside condition (e.g. birads number). Conditions are all -1 if unconditioned.
                try:
                    (
                        data,
                        conditions,
                        _,
                    ) = data
                except:
                    data, conditions, _, _ = data

                # Compute the actual batch size (not from config!) as last batch might be smaller than.
                b_size = data.size(0)

                # Format batch (fake and real), get images and, optionally, corresponding conditional GAN inputs
                real_images = data.to(self.device)

                real_conditions = None
                fake_conditions = None
                if self.config.conditional:
                    real_conditions = conditions.to(self.device)
                    # generate fake conditions
                    fake_images, fake_conditions = self._netG_forward_pass(
                        b_size=b_size
                    )
                else:
                    # Generate fake image batch with G (without condition)
                    fake_images, _ = self._netG_forward_pass(b_size=b_size)

                # We start by updating the discriminator
                # Reset the gradient of the discriminator of previous training iterations
                self.netD.zero_grad()

                # Perform a forward backward training step for D with optimizer weight update for real and fake data
                (
                    output_real_D,
                    D_x,
                    output_fake_D,
                    D_G_z1,
                    errD,
                    gradient_penalty,
                ) = self._netD_update(
                    netD=self.netD,
                    optimizerD=self.optimizerD,
                    real_images=real_images,
                    fake_images=fake_images.detach(),
                    epoch=epoch,
                    are_outputs_logits=False,
                    real_conditions=real_conditions,
                    fake_conditions=fake_conditions,
                )

                if self.config.pretrain_classifier:
                    self.netD2.zero_grad()
                    are_outputs_logits = (
                        True if self.config.model_name == "swin_transformer" else False
                    )
                    (
                        output_real_D2,
                        D2_x,
                        output_fake_D2,
                        D2_G_z1,
                        errD2,
                        gradient_penalty,
                    ) = self._netD_update(
                        netD=self.netD2,
                        optimizerD=self.optimizerD2,
                        real_images=real_images,
                        fake_images=fake_images.detach(),
                        epoch=epoch,
                        are_outputs_logits=are_outputs_logits,
                        real_conditions=real_conditions,
                        fake_conditions=fake_conditions,
                    )

                errG = None
                errG_D2 = None
                # After updating the discriminator, we now update the generator
                if not d_iteration == self.config.d_iters_per_g_update:
                    # We update D n times per G update. n = self.config.d_iters_per_g_update
                    d_iteration = d_iteration + 1

                else:

                    # Reset d_iteration to zero.
                    d_iteration = 0

                    # Reset to zero as previous gradient should have already been used to update the generator network
                    self.netG.zero_grad()

                    # (optional) Incomment if you want to generate new fake images as previous ones are already incorporated in D's gradient update
                    # fake_images, fake_conditions = self.generate_during_training(b_size=b_size)

                    # Perform a forward backward training step for G with optimizer weight update including a second
                    # output prediction by D to get bigger gradients as D has been already updated on this fake image batch.
                    (
                        output_fake_2_D1,
                        D_G_z2,
                        errG,
                        output_fake_2_D2,
                        D2_G_z2,
                        errG_D2,
                    ) = self.handle_G_updates(
                        iteration=i,
                        fake_images=fake_images,
                        fake_conditions=fake_conditions,
                        epoch=epoch,
                        b_size=b_size,
                        is_D2_using_new_fakes=False,  # TODO: Try is_D2_using_new_fakes out and see if it works better
                    )

                    # Save G Losses for plotting later
                    G_losses.append(errG.item())
                    # Update the running loss of G which is used in visualization
                    running_loss_of_generator += errG.item()

                    if self.config.pretrain_classifier:
                        G2_losses.append(errG_D2.item())
                        running_loss_of_generator_D2 += errG_D2.item()

                # Save D Losses for plotting later
                D_losses.append(errD.item())
                # Update the running loss which is used in visualization
                running_loss_of_discriminator += errD.item()

                if self.config.pretrain_classifier:
                    D2_losses.append(errD2.item())
                    running_loss_of_discriminator2 += errD2.item()

                # Accuracy of D1 (and D2) if they predict real/fake of synthetic images from G
                (
                    running_real_D_acc,
                    running_fake_D_acc,
                    current_real_D_acc,
                    current_fake_D_acc,
                    running_real_D2_acc,
                    running_fake_D2_acc,
                    current_real_acc_2,
                    current_fake_acc_2,
                ) = self.compute_discriminator_accuracy(
                    output_real_D=output_real_D,
                    running_real_D_acc=running_real_D_acc,
                    output_fake_D=output_fake_D,
                    running_fake_D_acc=running_fake_D_acc,
                    output_real_D2=None
                    if not self.config.pretrain_classifier
                    else output_real_D2,
                    running_real_D2_acc=None
                    if not self.config.pretrain_classifier
                    else running_real_D2_acc,
                    output_fake_D2=None
                    if not self.config.pretrain_classifier
                    else output_fake_D2,
                    running_fake_D2_acc=None
                    if not self.config.pretrain_classifier
                    else running_fake_D2_acc,
                )

                # Output training stats on each iteration length threshold
                if i % self.config.num_iterations_between_prints == 0:
                    self.periodic_training_console_log(
                        epoch=epoch,
                        iteration=i,
                        errD=errD,
                        errG=errG,
                        D_x=D_x,
                        D_G_z1=D_G_z1,
                        D_G_z2=D_G_z2,
                        current_real_D_acc=current_real_D_acc,
                        current_fake_D_acc=current_fake_D_acc,
                        errD2=None if not self.config.pretrain_classifier else errD2,
                        errG_D2=None
                        if not self.config.pretrain_classifier
                        else errG_D2,
                        D2_x=None if not self.config.pretrain_classifier else D2_x,
                        D2_G_z1=None
                        if not self.config.pretrain_classifier
                        else D2_G_z1,
                        D2_G_z2=None
                        if not self.config.pretrain_classifier
                        else D2_G_z2,
                        current_real_acc_2=None
                        if not self.config.pretrain_classifier
                        else current_real_acc_2,
                        current_fake_acc_2=None
                        if not self.config.pretrain_classifier
                        else current_fake_acc_2,
                        gradient_penalty=gradient_penalty,
                    )
                    self.periodic_visualization_log(
                        epoch=epoch,
                        iteration=i,
                        running_loss_of_generator=running_loss_of_generator,
                        running_loss_of_discriminator=running_loss_of_discriminator,
                        running_real_discriminator_accuracy=running_real_D_acc,
                        running_fake_discriminator_accuracy=running_fake_D_acc,
                        running_loss_of_generator_D2=None
                        if not self.config.pretrain_classifier
                        else running_loss_of_generator_D2,
                        running_loss_of_discriminator2=None
                        if not self.config.pretrain_classifier
                        else running_loss_of_discriminator2,
                        running_real_discriminator2_accuracy=None
                        if not self.config.pretrain_classifier
                        else running_real_D2_acc,
                        running_fake_discriminator2_accuracy=None
                        if not self.config.pretrain_classifier
                        else running_fake_D2_acc,
                    )

                    # Reset the running losses and accuracies
                    (
                        running_loss_of_generator,
                        running_loss_of_discriminator,
                        running_real_D_acc,
                        running_fake_D_acc,
                        _,
                        _,
                    ) = init_running_losses(init_value=0.0)

                    if self.config.pretrain_classifier:
                        (
                            running_loss_of_discriminator2,
                            running_loss_of_generator_D2,
                            running_real_D2_acc,
                            running_fake_D2_acc,
                            _,
                            _,
                        ) = init_running_losses(init_value=0.0)

                iters += 1
            self.visualization_utils.plot_losses(
                D_losses=D_losses,
                D2_losses=D2_losses,
                G_losses=G_losses,
                G2_losses=G2_losses,
            )
            if (
                epoch % self.config.num_epochs_between_gan_storage == 0
                and epoch >= self.config.num_epochs_before_gan_storage
            ):
                # Save on each {self.config.num_epochs_between_gan_storage}'th epoch starting at epoch {self.config.num_epochs_before_gan_storage}.
                save_model(
                    netD=self.netD,
                    optimizerD=self.optimizerD,
                    netG=self.netG,
                    optimizerG=self.optimizerG,
                    netD2=None if not self.config.pretrain_classifier else self.netD2,
                    optimizerD2=None
                    if not self.config.pretrain_classifier
                    else self.optimizerD2,
                    output_model_dir=self.config.output_model_dir,
                    epoch_number=epoch,
                )

            save_model(
                netD=self.netD,
                optimizerD=self.optimizerD,
                netG=self.netG,
                optimizerG=self.optimizerG,
                netD2=None if not self.config.pretrain_classifier else self.netD2,
                optimizerD2=None
                if not self.config.pretrain_classifier
                else self.optimizerD2,
                output_model_dir=self.config.output_model_dir,
            )
