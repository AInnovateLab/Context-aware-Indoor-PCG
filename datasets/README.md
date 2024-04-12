# Data pipeline

You can download the data used in our paper directly. Or you can try to reproduce the data from scratch.

The final data structure of `datasets/` is as follows:
```
datasets
├── data_transform.py
├── nr3d
│   ├── nr3d.csv   (optional)
│   └── nr3d_sa.csv
├── README.md
├── scannet -> /PATH/TO/SCANNET
│   ├── instruct-insertion
│   ├── scannetv2-labels.combined.tsv
│   ├── scans
│   ├── scans_test
│   └── tasks
└── sr3d
│   ├── sr3d+.csv   (optional)
    └── sr3d+_sa.csv
```

## Download ScanNet data

First, go to [ScanNet Repo](https://github.com/ScanNet/ScanNet) and get their downloading script as `instruct-insertion/scripts/download_scannet.py`. We did not include the complete script in our repo because ScanNet is released under their Term of Use, not MIT license.

We modified the script to remove unnecessary data. Make sure the following lines in `instruct-insertion/scripts/download_scannet.py`:
```python
FILETYPES = [
    ".aggregation.json",
    # ".sens",
    ".txt",
    # "_vh_clean.ply",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean_2.ply",
    # "_vh_clean.segs.json",
    # "_vh_clean.aggregation.json",
    "_vh_clean_2.labels.ply",
    # "_2d-instance.zip",
    # "_2d-instance-filt.zip",
    # "_2d-label.zip",
    # "_2d-label-filt.zip",
]
```

Since the [ScanNet](http://www.scan-net.org/) dataset is large, we recommend you to create `datasets/scannet` as a soft link to somewhere in the HDD before running our downloading script.

```shell
python instruct-insertion/scripts/download_scannet.py -o datasets/scannet
```

After the ScanNet is downloaded, preprocess the data by running:
```shell
bash instruct-insertion/scripts/preprocess_scannet_data.sh
```

## Context-aware Indoor PCG data

### Donwload from Google Drive

The `Sr3D-SA` and `Nr3D-SA` datasets can be downloaded from the [Anonymous Google Drive](https://dl.orangedox.com/SgTDvP1fLx5PCvNAFC).

Put them under `datasets/sr3d` and `datasets/nr3d` respectively.

### Build from source

If you want to build the data from source, you should set up [ReferIt3D](https://referit3d.github.io/) dataset and OpenAI API Token first.

Download ReferIt3D dataset:
```shell
pushd datasets
# Download Nr3D
gdown https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view?usp=drive_link -O nr3d/nr3d.csv
# Download Sr3D+
gdown https://drive.google.com/file/d/1kcxscHp8yA_iKLT3BdTNghlY3SBfKAQI/view?usp=drive_link -O sr3d/sr3d+.csv
popd
```

Check [datasets/data_transform.py](data_transform.py) for more details.
