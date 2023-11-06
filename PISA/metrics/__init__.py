import pathlib

LOCAL_METRIC_PATHS = {
    "accuracy_with_ignore_label": str(pathlib.Path(__file__).parent / "accuracy.py"),
    "accuracy_top_k_with_ignore_label": str(
        pathlib.Path(__file__).parent / "accuracy_with_top_k.py"
    ),
    "loc_estimate": str(pathlib.Path(__file__).parent / "loc_estimate.py"),
    "loc_estimate_with_top_k": str(pathlib.Path(__file__).parent / "loc_estimate_with_top_k.py"),
    "pairwise_cd": str(pathlib.Path(__file__).parent / "cd.py"),
    "average": str(pathlib.Path(__file__).parent / "average.py"),
}
