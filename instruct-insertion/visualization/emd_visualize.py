import os
import pickle
import sys

import accelerate

target_folder = "fps_axisnorm_rr4_sr3d"

sys.path.append(os.path.join(os.getcwd(), ".."))
accelerator = accelerate.Accelerator()
device = accelerator.device

with open(os.path.join(target_folder, "avg_emd_of_classes.pkl"), "rb") as f:
    avg_emd_of_classes = pickle.load(f)

import numpy as np

for key, value in avg_emd_of_classes.items():
    avg_emd_of_classes[key] = value.cpu().numpy()

# print(avg_emd_of_classes)

with open(os.path.join(target_folder, "num_of_obj_in_classes.pkl"), "rb") as f:
    num_of_obj_in_classes = pickle.load(f)
print(num_of_obj_in_classes)
total = 0
for key, values in num_of_obj_in_classes.items():
    total += values
print(total)

from data.referit3d.in_out.neural_net_oriented import load_scan_related_data

SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global.pkl"
_, _, class_to_idx = load_scan_related_data(SCANNET_PKL_FILE)

total = sum(num_of_obj_in_classes[item] for item in num_of_obj_in_classes)
import copy

num_of_obj_in_classes_copy = copy.deepcopy(num_of_obj_in_classes)
# Calculate the frequency of each class
for key, value in num_of_obj_in_classes.items():
    num_of_obj_in_classes[key] = num_of_obj_in_classes[key] / total

import pandas as pd

# print(class_to_idx)
# Reverse to get idx_to_class
idx_to_class = {v: k for k, v in class_to_idx.items()}

# read accuracy
with open(os.path.join(target_folder, "cls_top1_correct.pkl"), "rb") as f:
    cls_top1_correct = pickle.load(f)

for key, value in cls_top1_correct.items():
    cls_top1_correct[key] = value

with open(os.path.join(target_folder, "cls_top5_correct.pkl"), "rb") as f:
    cls_top5_correct = pickle.load(f)

for key, value in cls_top5_correct.items():
    cls_top5_correct[key] = value

new_dict = {}
# Change the class into string
for key, value in avg_emd_of_classes.items():
    new_key = f"{idx_to_class[key]} ({str('%.1f'%(num_of_obj_in_classes[key] * 100))}%)"
    new_dict[new_key] = [
        str("%.4f" % value.astype(np.float64)),
        str("%.2f" % (cls_top1_correct[key] / num_of_obj_in_classes_copy[key] * 100)),
        str("%.2f" % (cls_top5_correct[key] / num_of_obj_in_classes_copy[key] * 100)),
    ]

# Merge the EMD values into a csv file
pd.DataFrame.from_dict(new_dict, orient="index").to_csv(os.path.join(target_folder, "emd.csv"))
