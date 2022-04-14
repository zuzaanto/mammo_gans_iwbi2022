"""
Credits: richardosuala (Richard Osuala)
"""

## Pypi imports
import matplotlib

matplotlib.use("agg")
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class VisualizationUtils:
    """This class contains different visualization mechanisms for the GAN training process such as tensorboard writers."""

    def __init__(
        self,
        num_iterations_per_epoch: int,
        num_iterations_between_prints: int,
        output_model_dir: str,
    ):
        """
        :param tensorboard_writer_dir: The location in the fs where tensorboard visualization files are stored.
        """
        # setup tensorboard writer
        self.output_model_dir = Path(output_model_dir)
        self.tensorboard_writer = SummaryWriter(f"{output_model_dir}/visualization")
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.num_iterations_between_prints = num_iterations_between_prints

    def generate_tensorboard_network_graph(
        self, neural_network, network_input_1, network_input_2=None
    ):
        if network_input_2 is None:
            # Add the unconditional model architecture to tensorboard
            self.tensorboard_writer.add_graph(
                model=neural_network, input_to_model=network_input_1
            )
        else:
            # Add the conditional model architecture to tensorboard
            self.tensorboard_writer.add_graph(
                model=neural_network, input_to_model=(network_input_1, network_input_2)
            )
        self.tensorboard_writer.close()

    def add_value_to_tensorboard_accuracy_diagram(
        self,
        epoch,
        iteration,
        running_real_discriminator_accuracy,
        running_fake_discriminator_accuracy,
        running_real_discriminator2_accuracy=None,
        running_fake_discriminator2_accuracy=None,
        main_tag: str = "discriminator accuracy",
    ):
        # Number of the current iteration in the global training process
        current_global_iteration: int = (
            epoch * self.num_iterations_per_epoch + iteration
        )

        # Initialize a dictionary that holds both D's fake and real accuracy
        accuracy_dictionary = {
            "real data": running_real_discriminator_accuracy
            / self.num_iterations_between_prints,
            "fake data": running_fake_discriminator_accuracy
            / self.num_iterations_between_prints,
        }

        if running_real_discriminator2_accuracy is not None:
            accuracy_dictionary["real data D2"] = (
                running_real_discriminator2_accuracy
                / self.num_iterations_between_prints
            )
        if running_fake_discriminator2_accuracy is not None:
            accuracy_dictionary["fake data D2"] = (
                running_fake_discriminator2_accuracy
                / self.num_iterations_between_prints
            )

        self.tensorboard_writer.add_scalars(
            main_tag=main_tag,
            tag_scalar_dict=accuracy_dictionary,
            global_step=current_global_iteration,
        )
        self.tensorboard_writer.close()

    def add_value_to_tensorboard_loss_diagram(
        self,
        epoch,
        iteration,
        running_loss_of_generator,
        running_loss_of_discriminator,
        running_loss_of_generator_D2=None,
        running_loss_of_discriminator2=None,
        main_tag: str = "training loss",
    ):
        # Number of the current iteration in the global training process
        current_global_iteration: int = (
            epoch * self.num_iterations_per_epoch + iteration
        )

        # Initialize a dictionary that holds both generator and discriminator losses
        loss_dictionary = {
            "generator": running_loss_of_generator / self.num_iterations_between_prints,
            "discriminator": running_loss_of_discriminator
            / self.num_iterations_between_prints,
        }

        if running_loss_of_discriminator2 is not None:
            loss_dictionary["disciminator2"] = (
                running_loss_of_discriminator2 / self.num_iterations_between_prints
            )
        if running_loss_of_generator_D2 is not None:
            loss_dictionary["generator_D2"] = (
                running_loss_of_generator_D2 / self.num_iterations_between_prints
            )

        # Write the loss_dictionary to the tensorboard
        self.tensorboard_writer.add_scalars(
            main_tag=main_tag,
            tag_scalar_dict=loss_dictionary,
            global_step=current_global_iteration,
        )
        self.tensorboard_writer.close()

    def add_generated_batch_to_tensorboard(
        self,
        neural_network,
        network_input_1,
        network_input_2,
        padding: int = 2,
        normalize: bool = True,
        img_name: str = "GAN-generated-img",
    ):
        with torch.no_grad():
            if network_input_2 is not None:
                generated_image = (
                    neural_network(network_input_1, network_input_2).detach().cpu()
                )
            else:
                generated_image = neural_network(network_input_1).detach().cpu()
            img_grid = vutils.make_grid(
                generated_image, padding=padding, normalize=normalize
            )
            self.tensorboard_writer.add_image(img_name, img_grid)

    def plot_losses(
        self,
        G_losses,
        D_losses,
        D2_losses=None,
        G2_losses=None,
        figsize=(20, 10),
        title: str = "Generator and Discriminator Loss During Training",
        is_shown: bool = False,
        is_saved: bool = True,
    ):
        fig = plt.figure(figsize=figsize)
        plt.title(title)
        plt.plot(G_losses, label="G losses")
        plt.plot(D_losses, label="D losses")
        if D2_losses is not None:
            plt.plot(D2_losses, label="D2 losses")
        if G2_losses is not None:
            plt.plot(G2_losses, label="G2 losses")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        if is_shown:
            plt.show()
        if is_saved:
            fig.savefig(
                str((self.output_model_dir / "training_progress.png").resolve()),
                dpi=fig.dpi,
            )
        plt.close()
