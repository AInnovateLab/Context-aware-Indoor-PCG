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
│   ├── PISA
│   ├── scannetv2-labels.combined.tsv
│   ├── scans
│   ├── scans_test
│   └── tasks
└── sr3d
│   ├── sr3d+.csv   (optional)
    └── sr3d+_sa.csv
```

## Download ScanNet data

The [ScanNet](http://www.scan-net.org/) dataset is large.
We recommend you to create `datasets/scannet` as a soft link to somewhere in the HDD before runing the download script.

```shell
python PISA/scripts/download_scannet.py -o datasets/scannet
```

After the ScanNet is downloaded, preprocess the data by running:
```shell
bash PISA/scripts/preprocess_scannet_data.sh
```

## PISA data

### Donwload from Google Drive

The `Sr3D-SA` and `Nr3D-SA` datasets can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1URNnoJbmz35MSxbUocWHH4QWNH4_1XQS?usp=drive_link).

Or download using [gdown](https://github.com/wkentaro/gdown) tool:
```shell
pushd datasets
gdown https://drive.google.com/drive/folders/1URNnoJbmz35MSxbUocWHH4QWNH4_1XQS?usp=drive_link --folder
mv PISA/nr3d . && mv PISA/sr3d . && rm -d PISA
popd
```

### Build from source

If you want to build the data from source, you should set up [ReferIt3D] dataset and OpenAI API Token first.

Download ReferIt3D dataset:
```shell
pushd datasets
# Download Nr3D
gdown https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view?usp=drive_link -O nr3d/nr3d.csv
# Download Sr3D+
gdown https://drive.google.com/file/d/1kcxscHp8yA_iKLT3BdTNghlY3SBfKAQI/view?usp=drive_link -O sr3d/sr3d+.csv
popd
```

Check `datasets/data_transform.py` for more details.
