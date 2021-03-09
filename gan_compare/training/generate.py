import torch.optim as optim
import argparse
import cv2
import torch

from gan_compare.data_utils.utils import interval_mapping
from gan_compare.training import config # TODO instead save a json config along with the model and read it from there


def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name: supported: dcgan and lsgan"
    )
    parser.add_argument(
        "--image_size", type=int, required=True, help="Image size: 64 or 128"
    )
    parser.add_argument(
        "--model_checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="How many samples to generate"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Load model
    if args.model_name =="dcgan":
        if args.image_size == 64:
            from gan_compare.training.dcgan.res64.discriminator import Discriminator
            from gan_compare.training.dcgan.res64.generator import Generator
            netG = Generator(config.ngpu)
            netD = Discriminator(config.ngpu)
        elif args.image_size == 128:
            from gan_compare.training.dcgan.res128.discriminator import Discriminator
            from gan_compare.training.dcgan.res128.generator import Generator
            netD = Discriminator(config.ngpu, leakiness=config.leakiness, bias=False)
            netG = Generator(config.ngpu)
        else:
            raise ValueError("Unsupported image size. Supported sizes are 128 and 64.")
    elif args.model_name == "lsgan":
        # only 64x64 image resolution will be supported
        assert args.image_size == 64, "Wrong image size for LSGAN, change it to 64x64 before proceeding."
        
        from gan_compare.training.lsgan.discriminator import Discriminator
        from gan_compare.training.lsgan.generator import Generator

        netG = Generator()
        netD = Discriminator()
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=config.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    checkpoint = torch.load(args.model_checkpoint_path)
    netD.load_state_dict(checkpoint['discriminator'])
    netG.load_state_dict(checkpoint['generator'])
    optimizerD.load_state_dict(checkpoint['optim_discriminator'])
    optimizerG.load_state_dict(checkpoint['optim_generator'])

    netG.eval()
    netD.eval()
    
    for ind in range(args.num_samples):
        fixed_noise = torch.randn(args.image_size, config.nz, 1, 1)
        fake = netG(fixed_noise).detach().cpu().numpy()
        for j, img_ in enumerate(fake):
            img_ = interval_mapping(img_.transpose(1, 2, 0), -1., 1., 0, 255)
            img_ = img_.astype('uint8')
            cv2.imshow("sample", img_*2)
            k = cv2.waitKey()
            if k==27:    # Esc key to stop
                break
        print("Press any key to see the next batch. Press ESC to quit.")
        k = cv2.waitKey()
        if k==27:    # Esc key to stop
            break
    cv2.destroyAllWindows()
