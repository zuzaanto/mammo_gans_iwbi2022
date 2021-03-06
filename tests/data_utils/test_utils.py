from pathlib import Path

import numpy as np

from gan_compare.data_utils.utils import (
    generate_cbis_ddsm_metapoints,
    get_bcdr_bbox,
    parse_str_to_list_of_ints,
)


def test_parse_str_to_list_of_ints():
    test_string = "1 2 3 4 5 6 7"

    result = parse_str_to_list_of_ints(test_string)

    assert len(result) == 6  # This is wrong ofc


def test_get_bcdr_bbox():
    l_x = "1 1 50 101"
    l_y = "11 11 60 111"

    result = get_bcdr_bbox(l_x, l_y)

    assert result[0] == 1
    assert result[1] == 11
    assert result[2] == 100
    assert result[3] == 100


def test_generate_cbis_ddsm_metapoints():
    mask = np.zeros((1000, 1000))
    mask[20:500, 40:600] = 1
    image_id = "1234qwe"
    image_path = Path("/path/to/datasets/CBIS-DDSM/image.dcm")
    patient_id = "1287fdSA"
    roi_type = "mass"
    csv_metadata = {}
    csv_metadata["breast density"] = 3
    csv_metadata["assessment"] = 2
    csv_metadata["left or right breast"] = "R"
    csv_metadata["image view"] = "MLO"
    csv_metadata["pathology"] = "malignant"

    result, _ = generate_cbis_ddsm_metapoints(
        patch_id=0,
        mask=mask,
        image_id=image_id,
        patient_id=patient_id,
        csv_metadata=csv_metadata,
        image_path=image_path,
        roi_type=roi_type,
    )

    assert len(result) == 1
