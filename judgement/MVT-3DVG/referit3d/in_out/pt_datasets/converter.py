import pickle
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from functional import seq

from ...data_generation.nr3d import decode_stimulus_string
from ..scannet_scan import ScannetScan, ThreeDObject


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
        if hook_type != False:
            with open(data_path, "rb") as fp:
                self.hook_data = pickle.load(fp)

        if hook_type == "sa":
            # load the hooked data
            assert self.hook_type_fn_name is not None, "Hook type function name is required for SA."
            # hook_data_sa = [
            #     {
            #         "prompt": "",
            #         "objs": list of numpy of shape [P, 6],
            #         "ref": numpy of shape [P, 6],
            #         "stimulus_id": str,
            #         "class": int,
            #         "class_str": str,
            #         "pred_xyz_raw": numpy of shape [5, 3] located in norm axis,
            #         "pred_xyz": numpy of shape [5, P, 3] located in scene,
            #         "radius": numpy of shape [P, 1] with all same value,
            #     }
            # ]
            pass
        elif hook_type in ["po", False]:
            pass
        elif hook_type == "train":
            random.seed(42)
            self.hook_data = random.choices(self.hook_data, k=len(self.hook_data) // 2)
        else:
            raise ValueError(f"Hook type {hook_type} not found.")

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
            return references, scans
        elif self.hook_type == "po":
            return references, scans
        elif self.hook_type == "train":
            if self.hook_type_fn_name == "train_wo_augment":
                return dataset_train_wo_augment(self, references=references, scans=scans)
            elif self.hook_type_fn_name == "train_w_augment":
                return dataset_train_w_augment(self, references=references, scans=scans)
            else:
                raise ValueError(f"Hook type function {self.hook_type_fn_name} not found.")
        else:
            raise ValueError(f"Hook type {self.hook_type} not found.")

    def hook_getitem(
        self, stimulus_id: str, box_info: np.ndarray, samples: List[np.ndarray], target_pos: int
    ):
        # find hook_entry
        hook_entries = seq(self.hook_data).filter(lambda x: x["stimulus_id"] == stimulus_id).list()
        if len(hook_entries) == 0:
            target_sample = samples[target_pos]
            return {
                "hook_xyz": target_sample[..., :3],
                "hook_rgb": target_sample[..., 3:6],
            }
        # hook_entry = random.choice(hook_entries)
        hook_entry = hook_entries[0]

        ret = dict()
        if self.hook_type == False:
            raise RuntimeError("Should not execute here.")
        elif self.hook_type == "sa":
            ret.update(
                self.type_fn(self.hook_type_fn_name)(
                    self,
                    hook_entry=hook_entry,
                    box_info=box_info,
                    samples=samples,
                    target_pos=target_pos,
                )
            )
        elif self.hook_type == "po":
            ret.update(
                ol_point_e_only(
                    self,
                    hook_entry=hook_entry,
                    box_info=box_info,
                    samples=samples,
                    target_pos=target_pos,
                )
            )
        elif self.hook_type == "train":
            if self.hook_type_fn_name == "train_wo_augment":
                raise RuntimeError("Should not execute here.")
            elif self.hook_type_fn_name == "train_w_augment":
                ret.update(
                    getitem_train_w_augment(
                        self,
                        hook_entry=hook_entry,
                        box_info=box_info,
                        samples=samples,
                        target_pos=target_pos,
                    )
                )
            else:
                raise ValueError(f"Hook type function {self.hook_type_fn_name} not found.")
        else:
            raise ValueError(f"Hook type {self.hook_type} not found.")

        return ret

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

    def need_hook_dataset(self):
        return self.hook_type != False

    def need_hook_getitem(self):
        if self.hook_type == "train" and self.hook_type_fn_name == "train_wo_augment":
            return False
        return self.hook_type != False


@Converter.register_type_fn("rlos")
def _rlos(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    """(SA) Random Loc & Our Shape."""
    scene_bbox = {
        "max_x": box_info[: len(samples), 0].max(),
        "min_x": box_info[: len(samples), 0].min(),
        "max_y": box_info[: len(samples), 1].max(),
        "min_y": box_info[: len(samples), 1].min(),
        "max_z": box_info[: len(samples), 2].max(),
        "min_z": box_info[: len(samples), 2].min(),
    }
    random_loc = np.random.uniform(
        low=[scene_bbox["min_x"], scene_bbox["min_y"], scene_bbox["min_z"]],
        high=[scene_bbox["max_x"], scene_bbox["max_y"], scene_bbox["max_z"]],
        size=(3,),
    )  # [3,]
    hook_xyz = hook_entry["objs"][0][..., :3] * hook_entry["radius"] + random_loc[None]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": hook_entry["objs"][0][..., 3:6],
    }


@Converter.register_type_fn("rlgs")
def _rlgs(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    """(SA) Random Loc & GT Shape."""
    scene_bbox = {
        "max_x": box_info[: len(samples), 0].max(),
        "min_x": box_info[: len(samples), 0].min(),
        "max_y": box_info[: len(samples), 1].max(),
        "min_y": box_info[: len(samples), 1].min(),
        "max_z": box_info[: len(samples), 2].max(),
        "min_z": box_info[: len(samples), 2].min(),
    }
    random_loc = np.random.uniform(
        low=[scene_bbox["min_x"], scene_bbox["min_y"], scene_bbox["min_z"]],
        high=[scene_bbox["max_x"], scene_bbox["max_y"], scene_bbox["max_z"]],
        size=(3,),
    )  # [3,]
    GT_shape = samples[target_pos][..., :3]  # [P, 3]
    GT_shape -= box_info[target_pos, :3][None]  # [P, 3]
    hook_xyz = GT_shape + random_loc[None]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": samples[target_pos][..., 3:6],
    }


@Converter.register_type_fn("rlrs")
def _rlrs(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    """(SA) Random Loc & Random Shape."""
    hook_shape = np.random.uniform(
        low=-0.8, high=0.8, size=(len(hook_entry["radius"]), 3)
    )  # [P, 3]
    scene_bbox = {
        "max_x": box_info[: len(samples), 0].max(),
        "min_x": box_info[: len(samples), 0].min(),
        "max_y": box_info[: len(samples), 1].max(),
        "min_y": box_info[: len(samples), 1].min(),
        "max_z": box_info[: len(samples), 2].max(),
        "min_z": box_info[: len(samples), 2].min(),
    }
    random_loc = np.random.uniform(
        low=[scene_bbox["min_x"], scene_bbox["min_y"], scene_bbox["min_z"]],
        high=[scene_bbox["max_x"], scene_bbox["max_y"], scene_bbox["max_z"]],
        size=(3,),
    )  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + random_loc[None]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": hook_entry["objs"][0][..., 3:6],
    }


@Converter.register_type_fn("glos")
def _glos(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    """(SA) GT Loc & Our Shape."""
    hook_shape = hook_entry["objs"][0][..., :3]  # [P, 3]
    GT_loc = box_info[target_pos, :3]  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + GT_loc[None]  # [P, 3]
    # radius for pointe only model
    ## radius = box_info[target_pos, 3] ** (1 / 3)
    ## hook_xyz = hook_shape * radius + GT_loc[None]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": hook_entry["objs"][0][..., 3:6],
    }


@Converter.register_type_fn("glrs")
def _glrs(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    """(SA) GT Loc & Random Shape."""
    hook_shape = np.random.uniform(low=-0.8, high=0.8, size=(1024, 3))  # [P, 3]
    GT_loc = box_info[target_pos, :3]  # [3,]
    hook_xyz = hook_shape * hook_entry["radius"] + GT_loc[None]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": hook_entry["objs"][0][..., 3:6],
    }


@Converter.register_type_fn("olgs")
def _olgs(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    """(SA) Our Loc & GT Shape."""
    GT_shape = samples[target_pos][..., :3]  # [P, 3]
    GT_shape -= box_info[target_pos, :3][None]  # [P, 3]
    hook_xyz = GT_shape + hook_entry["pred_xyz_raw"][0]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": samples[target_pos][..., 3:6],
    }


def ol_point_e_only(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    """(PO) Our Loc & Point-E Shape."""
    hook_shape_po = hook_entry["objs"][0][..., :3]  # [P, 3]
    # radius for pointe only model
    radius = box_info[target_pos, 3] ** (1 / 3)
    hook_xyz = hook_shape_po * radius + hook_entry["pred_xyz_raw"][0]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": hook_entry["objs"][0][..., 3:6],
    }


def dataset_train_wo_augment(
    converter: Converter, references: pd.DataFrame, scans: Dict[str, ScannetScan]
) -> Tuple[pd.DataFrame, Dict[str, ScannetScan]]:
    """(Train) Without Augmentation."""
    data = converter.hook_data
    assert data is not None
    stimulus_ids = [datum["stimulus_id"] for datum in data]
    references = references[~references["stimulus_id"].isin(stimulus_ids)]
    return references, scans


def dataset_train_w_augment(
    converter: Converter, references: pd.DataFrame, scans: Dict[str, ScannetScan]
) -> Tuple[pd.DataFrame, Dict[str, ScannetScan]]:
    """(Train) With Augmentation."""
    # Do nothing.
    return references, scans


def getitem_train_w_augment(
    converter: Converter,
    hook_entry: Dict[str, Any],
    box_info: np.ndarray,
    samples: List[np.ndarray],
    target_pos: int,
):
    hook_shape = hook_entry["objs"][0][..., :3]  # [P, 3]
    GT_loc = box_info[target_pos, :3]  # [3,]
    radius = box_info[target_pos, 3] ** (1 / 3)
    hook_xyz = hook_shape * radius + GT_loc[None]  # [P, 3]
    # hook_xyz = hook_shape * hook_entry["radius"] + GT_loc[None]  # [P, 3]
    return {
        "hook_xyz": hook_xyz,
        "hook_rgb": hook_entry["objs"][0][..., 3:6],
    }
