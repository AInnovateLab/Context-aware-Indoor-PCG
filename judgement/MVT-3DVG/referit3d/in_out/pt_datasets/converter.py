import pickle
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..scannet_scan import ScannetScan


class Converter:
    _hook_functions = dict()

    def __init__(
        self,
        data_path,
        hook_type: Literal[False, "sa", "po", "train"],
        hook_type_fn_name: Optional[str] = None,
    ):
        self.hook_type = hook_type
        self.hook_type_fn_name = hook_type_fn_name
        if hook_type == "sa":
            # load the hooked data
            assert self.hook_type_fn_name is not None, "Hook type function name is required for SA."
            with open(
                data_path,
                "rb",
            ) as fp:
                self.hook_data = pickle.load(fp)

        elif hook_type == "po":
            with open(
                data_path,
                "rb",
            ) as fp:
                self.hook_data_po = pickle.load(fp)

        elif hook_type == "train":
            # TODO - trainset modifying hook
            pass

    def modify_dataset(
        self,
        references: pd.DataFrame,
        scans: Dict[str, ScannetScan],
    ) -> Tuple[pd.DataFrame, Dict[str, ScannetScan]]:
        """
        Args:
            references (pd.DataFrame): The references dataframe.
            scans (Dict[str, ScannetScan]): The scans dictionary.
        """
        if self.hook_type == False:
            # Do nothing.
            return references, scans
        elif self.hook_type == "sa":
            assert "utterance" in self.references.columns
            return self.type_fn(self.hook_type_fn_name)(self, references=references, scans=scans)
        elif self.hook_type == "po":
            # TODO
            pass
        elif self.hook_type == "train":
            # TODO
            pass
        else:
            raise ValueError(f"Hook type {self.hook_type} not found.")

    @classmethod
    def register_type_fn(cls, type_fn_hook_name: str):
        def decorator(func):
            cls._hook_functions[type_fn_hook_name] = func
            return func

        return decorator

    @classmethod
    def type_fn(cls, type_fn_hook_name: str):
        assert (
            type_fn_hook_name in cls._hook_functions
        ), f"Type function {type_fn_hook_name} not found."
        return cls._hook_functions[type_fn_hook_name]


# FIXME: All functions below is wrong.
@Converter.register_type_fn("rlos")
def _rlos(
    converter: Converter, references: pd.DataFrame, scans: Dict[str, ScannetScan]
) -> Tuple[pd.DataFrame, Dict[str, ScannetScan]]:
    # random loc + ours shape
    # first get the bbox of the entire scene

    # scene_bbox = {
    #     "max_x": box_info[: len(context), 0].max(),
    #     "min_x": box_info[: len(context), 0].min(),
    #     "max_y": box_info[: len(context), 1].max(),
    #     "min_y": box_info[: len(context), 1].min(),
    #     "max_z": box_info[: len(context), 2].max(),
    #     "min_z": box_info[: len(context), 2].min(),
    # }
    # random_xyz = np.random.uniform(
    #     low=[scene_bbox["min_x"], scene_bbox["min_y"], scene_bbox["min_z"]],
    #     high=[scene_bbox["max_x"], scene_bbox["max_y"], scene_bbox["max_z"]],
    #     size=(3,),
    # )  # [3,]
    # hook_xyz = hook_entry["objs"][0][..., :3] * hook_entry["radius"] + random_xyz[None]  # [P, 3]
    # return hook_xyz

    # TODO: modify the references and scans
    return references, scans


@Converter.register_type_fn("rlgs")
def _rlgs(converter: Converter, box_info, context):
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


@Converter.register_type_fn("rlrs")
def _rlrs():
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


@Converter.register_type_fn("glos")
def _glos(box_info):
    # GT Loc & Ours shape
    hook_shape = hook_entry["objs"][0][..., :3]  # [P, 3]
    GT_loc = box_info[target_pos, :3]  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + GT_loc[None]  # [P, 3]
    # radius for pointe only model
    radius = box_info[target_pos, 3] ** (1 / 3)
    hook_xyz = hook_shape * radius + GT_loc[None]  # [P, 3]
    return hook_xyz


@Converter.register_type_fn("glrs")
def _glrs(box_info):
    # GT Loc & Random shape
    hook_shape = np.random.uniform(low=-0.8, high=0.8, size=(1024, 3))  # [P, 3]
    GT_loc = box_info[target_pos, :3]  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + GT_loc[None]  # [P, 3]
    return hook_xyz


@Converter.register_type_fn("olgs")
def _olgs(box_info):
    # Ours Loc & GT shape
    GT_shape = samples[target_pos][..., :3]  # [P, 3]
    GT_shape -= box_info[target_pos, :3][None]  # [P, 3]
    hook_xyz = GT_shape + hook_entry["pred_xyz_raw"][0]  # [P, 3]
    return hook_xyz


@Converter.register_type_fn("ol_point_e_only")
def _ol_point_e_only(box_info):
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
