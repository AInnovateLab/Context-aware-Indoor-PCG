import pathlib

LOCAL_METRIC_PATHS = {
    "accuracy_with_ignore_label": str(pathlib.Path(__file__).parent / "accuracy.py"),
    "loc_estimate": str(pathlib.Path(__file__).parent / "loc_estimate.py"),
    "point_e_pc_jsd": str(pathlib.Path(__file__).parent / "jsd.py"),
}
