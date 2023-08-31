import logging
import os.path as osp
import sys
from typing import TYPE_CHECKING

import coloredlogs

if TYPE_CHECKING:
    from . import PathLike


def init_logger(
    mod_name: str = None,
    level: int = logging.INFO,
    log_file: "PathLike" = None,
    log_dir: "PathLike" = None,
):
    assert (
        log_file is not None or log_dir is not None
    ), "Either log_file or log_dir must be provided."
    logger = logging.getLogger(mod_name)
    fmt_str = "[%(levelname)s] %(asctime)s - %(message)s"
    coloredlogs.install(level=level, logger=logger, fmt=fmt_str)

    # Add logging to file handler
    if log_file is not None or log_dir is not None:
        filepath = log_file if log_file is not None else osp.join(log_dir, "log.txt")
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt_str))
        logger.addHandler(file_handler)

    return logger


def get_logger(mod_name: str = None):
    return logging.getLogger(mod_name)
