# Instruct-Replacement

Pull the repo with submodules:
```shell
git clone --recurse-submodules git@github.com:MRTater/Instruct-Replacement.git
```

Update submodules if not included:
```shell
git submodule update --init --recursive
```

Update only submodules if already exists:
```shell
git submodule update --recursive --remote
```

## Dataset

Create soft links to the dataset folder in the root directory of the project.

```bash
mkdir -p datasets
ln -s /media/data1/share/datasets/scannet_referit3d datasets/scannet
ln -s /media/data1/share/datasets/nr3d datasets/nr3d
ln -s /media/data1/share/datasets/sr3d datasets/sr3d
```

## Setup
Create environment:
```
conda create -n instruct-insertion python=3.9
conda activate instruct-insertion
pip install -r requirements.txt
```

Next, install cpp extensions for pointnet:
```shell
conda install -c conda-forge cudatoolkit-dev
cd instruct-insertion/openpoints/cpp/pointnet2_batch
python setup.py install
```
