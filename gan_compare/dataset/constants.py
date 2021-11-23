BIRADS_DICT = {
    "2": 1,
    "3": 2,
    "4a": 3,
    "4b": 4,
    "4c": 5,
    "5": 6,
    "6": 7,
}

BCDR_BIRADS_DICT = {
    "Benign": 3,
    "Malign": 6,
}

DENSITY_DICT = {
    "1": 0.0,
    "2": 0.33,
    "3": 0.67,
    "4": 1.0,
}

BCDR_SUBDIRECTORIES = {
    "d01": "BCDR-D01_dataset",
    "d02": "BCDR-D02_dataset",
    "f01": "BCDR-F01_dataset",
    "f02": "BCDR-F02_dataset",
    "f03": "BCDR-F03",
}

BCDR_HEALTHY_SUBDIRECTORIES = {
    "dn01": "BCDR-DN01_dataset",
}

BCDR_VIEW_DICT = {
    1: {"laterality": "R", "view": "CC"},
    2: {"laterality": "L", "view": "CC"},
    3: {"laterality": "R", "view": "MLO"},
    4: {"laterality": "L", "view": "MLO"},
}
