from time import time

# Birads range used as condition input into cGAN
birads_min = 2
birads_max = 6

# The number of condition labels for input into cGAN (i.e. BI-RADS 2,3,4,5,6)
# Note: torch nn.Embedding start to count the num_embeddings at index 0 --> [0,birads_max+1] --> nn.Embedding(birads_max+1,dim)
n_cond = birads_max + 1

# l2 regularization in discriminator
weight_decay = 0.5

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 16

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Whether to use least square loss
use_lsgan_loss = False

# Leakiness for ReLUs
leakiness = 0.3

# Number of channels in the training images. For color images this is 3
nc = 2

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# The folder in which the model checkpoint wil be stored.
output_model_dir = f"model_checkpoints/training_{time()}/{image_size}/"

# The path to a model checkpoint that is too be loaded.
pretrained_model_path = f"model_checkpoints/{image_size}/model.pt"
