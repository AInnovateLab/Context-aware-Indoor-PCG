import os
import pickle
import sys

import accelerate
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), ".."))
accelerator = accelerate.Accelerator()
device = accelerator.device
from data.referit3d.in_out.neural_net_oriented import load_scan_related_data

target_folder = "fps_axisnorm_rr4_sr3d"
with open(os.path.join(target_folder, "all_data.pkl"), "rb") as f:
    all_data = pickle.load(f)

with open(
    "/home/lyy/workspace/Instruct-Replacement/tmp_link_saves/cache/train_data_percentage.pkl", "rb"
) as f:
    train_data_percentage = pickle.load(f)

SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global.pkl"
_, _, class_to_idx = load_scan_related_data(SCANNET_PKL_FILE)
# read accuracy
with open(os.path.join(target_folder, "cls_top1_correct.pkl"), "rb") as f:
    cls_top1_correct = pickle.load(f)

for key, value in cls_top1_correct.items():
    cls_top1_correct[key] = value

with open(os.path.join(target_folder, "cls_top5_correct.pkl"), "rb") as f:
    cls_top5_correct = pickle.load(f)

for key, value in cls_top5_correct.items():
    cls_top5_correct[key] = value

with open(os.path.join(target_folder, "num_of_obj_in_classes.pkl"), "rb") as f:
    num_of_obj_in_classes = pickle.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
one_nn = {}
one_nna = {}
for key, value in all_data.items():
    if key == "one_nn":
        for k, v in value.items():
            new_key = (
                f"{idx_to_class[k]}" + f"({train_data_percentage[idx_to_class[k]] * 100:.2f}\\%)"
            )
            one_nn[new_key] = [
                round(min(v).item(), 3),
                str("%.2f" % (cls_top1_correct[k] / num_of_obj_in_classes[k] * 100)),
                str("%.2f" % (cls_top5_correct[k] / num_of_obj_in_classes[k] * 100)),
            ]


def real_key(kv):
    k, v = kv
    return k.split("(")[0]


key = lambda kv: train_data_percentage[real_key(kv)]
one_nn = dict(sorted(one_nn.items(), key=key, reverse=True))


pd.DataFrame.from_dict(one_nn, orient="index").to_csv("one_nn.csv")
# pd.DataFrame.from_dict(one_nna, orient="index").to_csv("one_nna.csv")
