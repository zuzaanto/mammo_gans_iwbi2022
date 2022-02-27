from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import yaml


def load_yaml(path: Union[str, Path]):
    with open(path, "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
            return {}


def save_yaml(path: Union[str, Path], data: dataclass):
    with open(path, "w") as yaml_file:
        return yaml.dump(asdict(data), yaml_file)
