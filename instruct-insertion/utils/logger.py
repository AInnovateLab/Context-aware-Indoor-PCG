import logging
import os.path as osp
import sys

from . import PathLike


def init_logger(
    log_file: PathLike = None,
    log_dir: PathLike = None,
    mod_name: str = None,
    level: int = logging.INFO,
    std_out: bool = True,
):
    assert (
        log_file is not None or log_dir is not None
    ), "Either log_file or log_dir must be provided."
    logger = logging.getLogger(mod_name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # Add logging to file handler
    filepath = log_file if log_file is not None else osp.join(log_dir, "log.txt")
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger


def get_logger(mod_name: str = None):
    return logging.getLogger(mod_name)
