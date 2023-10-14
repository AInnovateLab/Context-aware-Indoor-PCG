import os
import pickle
import sys

import accelerate

sys.path.append(os.path.join(os.getcwd(), ".."))
accelerator = accelerate.Accelerator()
device = accelerator.device

with open("avg_emd_of_classes.pkl", "rb") as f:
    avg_emd_of_classes = pickle.load(f)

import numpy as np

for key, value in avg_emd_of_classes.items():
    avg_emd_of_classes[key] = value.cpu().numpy()

# print(avg_emd_of_classes)

with open("num_of_obj_in_classes.pkl", "rb") as f:
    num_of_obj_in_classes = pickle.load(f)

from data.referit3d.in_out.neural_net_oriented import load_scan_related_data

SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global.pkl"
_, _, class_to_idx = load_scan_related_data(SCANNET_PKL_FILE)

total = sum(num_of_obj_in_classes[item] for item in num_of_obj_in_classes)
# Calculate the frequency of each class
for key, value in num_of_obj_in_classes.items():
    num_of_obj_in_classes[key] = num_of_obj_in_classes[key] / total

import pandas as pd

# print(class_to_idx)
# Reverse to get idx_to_class
idx_to_class = {v: k for k, v in class_to_idx.items()}

new_emd_dict = {}
# Change the class into string
for key, value in avg_emd_of_classes.items():
    new_key = f"{idx_to_class[key]} ({str('%.1f'%(num_of_obj_in_classes[key] * 100))}%)"
    new_emd_dict[new_key] = str("%.4f" % value.astype(np.float64))

# Merge the EMD values into a csv file
pd.DataFrame.from_dict(new_emd_dict, orient="index").to_csv("emd.csv")

# save the plot
# plt.savefig("example.png", )
