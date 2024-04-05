import pickle

import numpy as np


class Converter:
    def __init__(
        self,
        hook,
        data_path,
    ):
        # self.args = args

        # TODO - references?
        # self.hook_sa = hook_sa
        # self.hook_po = hook_po
        # self.hook_train = hook_train

        # TODO - hook redefined
        if hook == "sa":
            assert "utterance_generative" in self.references.columns
            print("Loading the hooked data...")
            # load the hooked data
            with open(
                data_path,
                "rb",
            ) as fp:
                self.hook_data = pickle.load(fp)

        elif hook == "po":
            with open(
                data_path,
                "rb",
            ) as fp:
                self.hook_data_po = pickle.load(fp)

        elif hook == "train":
            # TODO - trainset modifying hook
            pass

    def modify_trainset(self):
        pass

    def modify_testset(self, type_fn, *args):
        self.register_type_fn(type_fn)
        return self.type_fn(*args)

    def register_type_fn(self, type_fn_hook):
        assert type_fn_hook in [
            "_rlos",
            "_rlgs",
            "_rlrs",
            "_glos",
            "_glrs",
            "_olgs",
            "_ol_point_e_only",
        ]
        self.type_fn_hook = type_fn_hook
        # TODO - 如果不用if，怎么把字符串作为hook来注册函数啊？eval会不会被骂啊
        self.type_fn = eval(self.type_fn_hook)


def _rlos(self, box_info, context, hook_entry):
    # random loc + ours shape
    # first get the bbox of the entire scene
    scene_bbox = {
        "max_x": box_info[: len(context), 0].max(),
        "min_x": box_info[: len(context), 0].min(),
        "max_y": box_info[: len(context), 1].max(),
        "min_y": box_info[: len(context), 1].min(),
        "max_z": box_info[: len(context), 2].max(),
        "min_z": box_info[: len(context), 2].min(),
    }
    random_xyz = np.random.uniform(
        low=[scene_bbox["min_x"], scene_bbox["min_y"], scene_bbox["min_z"]],
        high=[scene_bbox["max_x"], scene_bbox["max_y"], scene_bbox["max_z"]],
        size=(3,),
    )  # [3,]
    hook_xyz = hook_entry["objs"][0][..., :3] * hook_entry["radius"] + random_xyz[None]  # [P, 3]
    return hook_xyz


def _rlgs(self, box_info, context):
    # random loc + GT shape
    # first get the bbox of the entire scene
    scene_bbox = {
        "max_x": box_info[: len(context), 0].max(),
        "min_x": box_info[: len(context), 0].min(),
        "max_y": box_info[: len(context), 1].max(),
        "min_y": box_info[: len(context), 1].min(),
        "max_z": box_info[: len(context), 2].max(),
        "min_z": box_info[: len(context), 2].min(),
    }
    random_xyz = np.random.uniform(
        low=[scene_bbox["min_x"], scene_bbox["min_y"], scene_bbox["min_z"]],
        high=[scene_bbox["max_x"], scene_bbox["max_y"], scene_bbox["max_z"]],
        size=(3,),
    )  # [3,]
    GT_shape = samples[target_pos][..., :3]  # [P, 3]
    GT_shape -= box_info[target_pos, :3][None]  # [P, 3]
    hook_xyz = GT_shape + random_xyz[None]  # [P, 3]
    return hook_xyz


def _rlrs(self):
    # random loc & random shape
    hook_shape = np.random.uniform(low=-0.8, high=0.8, size=(1024, 3))  # [P, 3]
    scene_bbox = {
        "max_x": box_info[: len(context), 0].max(),
        "min_x": box_info[: len(context), 0].min(),
        "max_y": box_info[: len(context), 1].max(),
        "min_y": box_info[: len(context), 1].min(),
        "max_z": box_info[: len(context), 2].max(),
        "min_z": box_info[: len(context), 2].min(),
    }
    random_xyz = np.random.uniform(
        low=[scene_bbox["min_x"], scene_bbox["min_y"], scene_bbox["min_z"]],
        high=[scene_bbox["max_x"], scene_bbox["max_y"], scene_bbox["max_z"]],
        size=(3,),
    )  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + random_xyz[None]  # [P, 3]
    return hook_xyz


def _glos(self, box_info):
    # GT Loc & Ours shape
    hook_shape = hook_entry["objs"][0][..., :3]  # [P, 3]
    GT_loc = box_info[target_pos, :3]  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + GT_loc[None]  # [P, 3]
    # radius for pointe only model
    radius = box_info[target_pos, 3] ** (1 / 3)
    hook_xyz = hook_shape * radius + GT_loc[None]  # [P, 3]
    return hook_xyz


def _glrs(self, box_info):
    # GT Loc & Random shape
    hook_shape = np.random.uniform(low=-0.8, high=0.8, size=(1024, 3))  # [P, 3]
    GT_loc = box_info[target_pos, :3]  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + GT_loc[None]  # [P, 3]
    return hook_xyz


def _olgs(self, box_info):
    # Ours Loc & GT shape
    GT_shape = samples[target_pos][..., :3]  # [P, 3]
    GT_shape -= box_info[target_pos, :3][None]  # [P, 3]
    hook_xyz = GT_shape + hook_entry["pred_xyz_raw"][0]  # [P, 3]
    return hook_xyz


def _ol_point_e_only(self, box_info):
    for hook_entry_po in self.hook_data_po:
        if hook_entry_po["stimulus_id"] == stimulus_id and hook_entry_po["prompt"] == gtext:
            break
        else:
            raise ValueError("Cannot find the hooked data for this sample.")
    # Ours Loc & Point-E only shape
    hook_shape_po = hook_entry_po["objs"][0][..., :3]  # [P, 3]
    # radius for pointe only model
    radius = box_info[target_pos, 3] ** (1 / 3)
    hook_xyz = hook_shape_po * radius + hook_entry["pred_xyz_raw"][0]  # [P, 3]
    return hook_xyz
