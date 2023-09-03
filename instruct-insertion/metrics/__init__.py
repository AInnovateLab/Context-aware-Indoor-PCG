import pathlib

LOCAL_METRIC_PATHS = {
    "accuracy_with_ignore_label": str(pathlib.Path(__file__).parent / "accuracy.py"),
    "loc_estimate": str(pathlib.Path(__file__).parent / "loc_estimate.py"),
    "pairwise_cd": str(pathlib.Path(__file__).parent / "cd.py"),
}
