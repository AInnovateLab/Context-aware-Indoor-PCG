import json
import os
import os.path as osp
import pickle
from argparse import ArgumentTypeError
from typing import Any, Dict, Generator, List, TypeVar, Union

PathLike = Union[str, bytes, os.PathLike]

K = TypeVar("K")
V = TypeVar("V")


def read_dict_from_json(filepath: PathLike) -> Dict:
    with open(filepath) as fin:
        return json.load(fin)


def load_json(filepath: PathLike):
    with open(filepath) as fin:
        return json.load(fin)


def read_lines(filepath: PathLike) -> List[str]:
    with open(filepath) as fin:
        return [line.rstrip() for line in fin.readlines()]


def invert_dictionary(d: Dict[K, V]) -> Dict[V, K]:
    return {v: k for k, v in d.items()}


def immediate_subdirectories(top_dir: PathLike, full_path: bool = True) -> List[str]:
    """
    Returns the immediate subdirectories of a given directory.
    """
    dir_names = [name for name in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, name))]
    if full_path:
        dir_names = [osp.join(top_dir, name) for name in dir_names]
    return dir_names


def pickle_data(filepath: PathLike, *args):
    """
    Using pickle to save multiple python objects in a single file.
    """
    with open(filepath, "wb") as out_file:
        pickle.dump(len(args), out_file, protocol=5)
        for item in args:
            pickle.dump(item, out_file, protocol=5)


def unpickle_data(filepath: PathLike) -> Generator[Any, None, None]:
    """
    Restore data previously saved with pickle_data().

    Args:
        file_name (PathLike): filepath holding the pickled data.

    Returns:
        Generator: a generator of the objects saved in the file.
    """
    with open(filepath, "rb") as in_file:
        size = pickle.load(in_file)

        for _ in range(size):
            yield pickle.load(in_file)


def create_dir(dir_path: PathLike):
    """
    Creates a directory (or nested directories) if they don't exist.
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def str2bool(v: Union[str, bool]):
    """
    Boolean values for argparse.
    """
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
