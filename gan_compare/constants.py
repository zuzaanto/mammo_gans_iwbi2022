from gan_compare.dataset.inbreast_dataset import InbreastDataset
from gan_compare.dataset.bcdr_dataset import BCDRDataset


DATASET_DICT = {
    "bcdr": BCDRDataset,
    "inbreast": InbreastDataset,
}
DENSITIES = [1,2,3,4]
