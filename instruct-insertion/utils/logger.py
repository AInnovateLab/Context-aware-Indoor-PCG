import logging
import os.path as osp
import sys
from typing import TYPE_CHECKING

# NOTE: re-import
from accelerate.logging import get_logger  # noqa

if TYPE_CHECKING:
    from . import PathLike


def init_logger(
    level: int = logging.INFO,
    log_file: "PathLike" = None,
    log_dir: "PathLike" = None,
):
    handlers = list()
    handlers.append(logging.StreamHandler(sys.stdout))
    fmt_str = "[%(levelname)s] %(asctime)s - %(message)s"

    # Add logging to file handler
    if log_file is not None or log_dir is not None:
        filepath = log_file if log_file is not None else osp.join(log_dir, "log.txt")
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(logging.Formatter(fmt_str))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=fmt_str, handlers=handlers)
