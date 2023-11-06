# NOTE: reimport common utils
from data.utils import PathLike, create_dir, load_json, read_lines, str2bool


##########################
#                        #
#    training related    #
#                        #
##########################
def seed_everything(seed: int, deterministic: bool = True, warn_only: bool = True):
    import warnings

    warnings.warn(
        "This function is deprecated. Use `accelerate.utils.set_seed` instead.", DeprecationWarning
    )

    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
