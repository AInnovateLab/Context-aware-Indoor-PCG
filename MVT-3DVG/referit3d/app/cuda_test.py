import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print(torch.cuda.default_stream())
