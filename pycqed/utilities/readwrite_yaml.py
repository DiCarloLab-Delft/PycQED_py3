# -------------------------------------------
# Functions for import/export and formatting yaml.
# -------------------------------------------
import os
import yaml
from typing import Union
from pycqed.definitions import ROOT_DIR


# --- YAML Read/Write  ---

def get_yaml_file_path(filename: str) -> str:
    """Returns yaml file path as used by read/write functions."""
    return os.path.join(ROOT_DIR, filename)


def read_yaml(filename: str) -> Union[object, FileNotFoundError]:
    """Returns yaml after importing from YAML_ROOT + filename."""
    file_path = get_yaml_file_path(filename=filename)
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def write_yaml(filename: str, packable: object, make_file: bool = False, *args, **kwargs) -> bool:
    """Returns if file exists. Dumps config_dict to yaml file."""
    # Dump dictionary into yaml file
    file_path = get_yaml_file_path(filename=filename)
    if not make_file and not os.path.isfile(file_path):
        return False
    with open(file_path, 'w') as f:
        yaml.dump(packable, f, default_flow_style=False, allow_unicode=True, encoding=None, *args, **kwargs)
    return True

# ------------------------
