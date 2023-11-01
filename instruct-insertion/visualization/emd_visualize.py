import copy
import os
import pickle
import sys

import accelerate
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), ".."))
from data.referit3d.in_out.neural_net_oriented import load_scan_related_data

target_folder_list = []
target_folder_list.append("fps_axisnorm_rr4_sr3d")
# target_folder_list.append("fps_axisnorm_rr4")
# target_folder_list.append("fps_axisnorm")
# target_folder_list.append("fps")
# target_folder_list.append("baseline")

for target_folder in target_folder_list:
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    with open(os.path.join(target_folder, "all_data.pkl"), "rb") as f:
        all_data = pickle.load(f)

    # with open(os.path.join(target_folder, "avg_emd_of_classes.pkl"), "rb") as f:
    #     avg_emd_of_classes = pickle.load(f)

    # load data in all_data
    # mmd_emd = all_data["mmd_emd"]
    one_nn = all_data["one_nn"]
    num_of_obj_in_classes = all_data["num_of_obj_in_classes"]

    final_one_nn = 0
    total_num_class = 0
    for key, values in one_nn.items():
        # one_nn[key] = min(value)
        for value in values:
            total_num_class += 1
            final_one_nn += value.cpu().item()
    print(target_folder + ": ", final_one_nn / total_num_class * 10)
quit()

final_emd = 0
total_mmd_emd_num = 0
for key, values in mmd_emd.items():
    total_mmd_emd_num += len(values)
    for value in values:
        final_emd += value.cpu().item()
print(target_folder + ": " + final_emd / total_mmd_emd_num)


# for key, value in avg_emd_of_classes.items():
#     avg_emd_of_classes[key] = value.cpu().numpy()

# print(avg_emd_of_classes)

# with open(os.path.join(target_folder, "num_of_obj_in_classes.pkl"), "rb") as f:
#     num_of_obj_in_classes = pickle.load(f)
# print(num_of_obj_in_classes)
# total = 0
# for key, values in num_of_obj_in_classes.items():
#     total += values
# print(total)

# SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global.pkl"
SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global_small.pkl"
_, _, class_to_idx = load_scan_related_data(SCANNET_PKL_FILE)

total = sum(num_of_obj_in_classes[item] for item in num_of_obj_in_classes)

# num_of_obj_in_classes_copy = copy.deepcopy(num_of_obj_in_classes)
# Calculate the frequency of each class
# for key, value in num_of_obj_in_classes.items():
#     num_of_obj_in_classes[key] = num_of_obj_in_classes[key] / total


# Reverse to get idx_to_class
idx_to_class = {v: k for k, v in class_to_idx.items()}

# read accuracy
# with open(os.path.join(target_folder, "cls_top1_correct.pkl"), "rb") as f:
#     cls_top1_correct = pickle.load(f)

# for key, value in cls_top1_correct.items():
#     cls_top1_correct[key] = value

# with open(os.path.join(target_folder, "cls_top5_correct.pkl"), "rb") as f:
#     cls_top5_correct = pickle.load(f)

# for key, value in cls_top5_correct.items():
#     cls_top5_correct[key] = value

new_dict = {}
# Change the class into string
for key, value in one_nn.items():
    new_key = f"{idx_to_class[key]} ({str('%.1f'%(num_of_obj_in_classes[key] * 100))}%)"
    new_dict[new_key] = [
        str("%.4f" % value.astype(np.float64)),
        str()
        # str("%.2f" % (cls_top1_correct[key] / num_of_obj_in_classes_copy[key] * 100)),
        # str("%.2f" % (cls_top5_correct[key] / num_of_obj_in_classes_copy[key] * 100)),
    ]

# Merge the EMD values into a csv file
# pd.DataFrame.from_dict(new_dict, orient="index").to_csv(os.path.join(target_folder, "emd.csv"))
