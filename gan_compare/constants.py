from gan_compare.dataset.inbreast_dataset import InbreastDataset
from gan_compare.dataset.bcdr_dataset import BCDRDataset
from gan_compare.training.networks.classification.swin_transformer import SwinTransformer
from gan_compare.training.networks.classification.cnn import CNNNet

DATASET_DICT = {
    "bcdr": BCDRDataset,
    "inbreast": InbreastDataset,
}

DENSITIES = [1,2,3,4]

CLASSIFIERS_DICT = {
    "cnn": CNNNet,
    "swin_transformer": SwinTransformer,
}
