import logging
import os.path as osp
import sys
from typing import TYPE_CHECKING

# NOTE: re-import
from accelerate import Accelerator
from accelerate.logging import get_logger  # noqa
from coloredlogs import ColoredFormatter

from .misc import create_dir

if TYPE_CHECKING:
    from . import PathLike


def init_logger(
    accelerator: Accelerator,
    log_file: "PathLike" = None,
    log_dir: "PathLike" = None,
):
    fmt_str = "[%(levelname)s, %(name)s] %(asctime)s - %(message)s"
    datefmt_str = "%Y-%m-%d %H:%M:%S"
    handlers = list()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(ColoredFormatter(fmt_str, datefmt=datefmt_str))
    handlers.append(stdout_handler)

    # Add logging to file handler
    if accelerator.is_main_process:
        if log_file is not None or log_dir is not None:
            if log_file is not None:
                create_dir(osp.dirname(log_file))
                filepath = log_file
            elif log_dir is not None:
                create_dir(log_dir)
                filepath = osp.join(log_dir, "log.txt")
            file_handler = logging.FileHandler(filepath)
            file_handler.setFormatter(logging.Formatter(fmt_str, datefmt=datefmt_str))
            handlers.append(file_handler)

    logging.basicConfig(format=fmt_str, handlers=handlers, datefmt=datefmt_str)
