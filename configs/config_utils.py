import pickle
import shutil
import trimesh
from pathlib import Path

from yaml import safe_dump, safe_load


class ConfigParams:
    """
    A class for managing configuration parameters with default values.

    Attributes:
        config (dict): A dictionary holding configuration parameters.

    Methods:
        __init__(self, config, default_values): Initializes the configuration with default values for missing parameters.
        __getitem__(self, item): Allows retrieval of configuration parameters using dictionary-like indexing.
    """
    def __init__(self, config, default_values):
        self.config = config
        for item in default_values:
            if item not in config:
                self.config[item] = default_values[item]

    def __getitem__(self, item):
        return self.config[item]


def get_params_values(args, key, default=None):
    """
    Retrieves the value of a specified key from a dictionary, returning a default value if the key is not found.

    Parameters:
        args (dict): The dictionary from which to retrieve the value.
        key (str): The key for which to retrieve the value.
        default (any, optional): The default value to return if the key is not found. Defaults to None.

    Returns:
        The value associated with the specified key in the dictionary, or the default value if the key is not found.
    """
    if (key in args) and (args[key] is not None):
        return args[key]
    return default


def read_yaml(yaml_file):
    """
    Reads a YAML file and returns its contents as a dictionary.

    Parameters:
        yaml_file (str): The path to the YAML file to be read.

    Returns:
        dict: The YAML file contents as a dictionary.
    """
    with open(yaml_file, 'r') as config_file:
        yaml_dict = safe_load(config_file)
    return yaml_dict


def _sanitize_for_yaml_export(obj):
    """Convert Path / numpy scalars to plain Python types safe for ``yaml.safe_dump`` (no ``!!python/object``)."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if isinstance(obj, Path):
        return str(obj)
    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_yaml_export(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_yaml_export(v) for v in obj)
    return obj


def copy_yaml(config_file, save_name='config_file.yaml'):
    """
    Save training configuration next to checkpoints.

    - Pass a path to the **source YAML** (``str`` / ``Path``): copy the file byte-for-byte so
      quoting, empty keys, and section order match the original.
    - Pass a **dict** (e.g. merged runtime config): ``safe_dump`` after sanitizing; ``Path``
      values become strings.
    """
    if isinstance(config_file, (str, Path)):
        src = Path(config_file).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f'config yaml not found: {src}')
        with open(src, 'r', encoding='utf-8') as fh:
            yfile = safe_load(fh)
        literal_copy = True
    elif isinstance(config_file, dict):
        yfile = config_file
        literal_copy = False
        src = None
    else:
        raise TypeError(f'copy_yaml expects str, Path or dict, got {type(config_file)!r}')

    dest_dir = Path(yfile['CHECKPOINT']['save_dir'])
    base = Path(save_name)
    stem0 = base.stem
    suf0 = base.suffix if base.suffix else '.yaml'
    dest = dest_dir / save_name
    i = 1
    while dest.is_file():
        dest = dest_dir / f'{stem0}_{i}{suf0}'
        i += 1

    dest.parent.mkdir(parents=True, exist_ok=True)

    if literal_copy:
        shutil.copyfile(src, dest)
    else:
        with open(dest, 'w', encoding='utf-8') as outfile:
            safe_dump(
                _sanitize_for_yaml_export(config_file),
                outfile,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )


def get_template_mean_std(config):
    """
    Retrieves the mean and standard deviation of a dataset, along with its template mesh, based on configuration.

    Parameters:
        config (dict): The configuration dictionary specifying the machine and dataset to be used.

    Returns:
        tuple: A tuple containing the template mesh (as a trimesh object), the mean, the standard deviation of
        the dataset and a multiplier to transform data set mesh dimensions to mm.
    """
    DATASET_INFO = read_yaml("data/implemented_datasets.yaml")[config['MACHINE']][
        config['DATASETS']['eval']['dataset']]
    mean_std_file = DATASET_INFO['mean_std_file']
    with open(mean_std_file, 'rb') as handle:
        mean_std = pickle.load(handle, encoding='latin1')
    mean = mean_std['mean']  # .numpy()
    std = mean_std['std']  # .numpy()  # + h
    mm_mult = mean_std['mm_mult']
    template_path = DATASET_INFO['template']
    mesh = trimesh.load(template_path)
    return mesh, mean, std, mm_mult
