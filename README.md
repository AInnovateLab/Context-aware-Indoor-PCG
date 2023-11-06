# PISA: Point Cloud based Instructed Scene Augmentation

## Install Openpoints

Pull the repo with submodules: (TODO: anomonous)
```shell
git clone --recurse-submodules git@github.com:MRTater/PISA.git
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
```shell
conda create -n pisa python=3.9
conda activate pisa
conda install -c conda-forge cudatoolkit-dev=11.7
pip install -r requirements.txt
```

If you want to cauculate MMD and 1-NNA, please run the following command:
```shell
cd PISA/openpoints/cpp/emd
python setup.py install
```

Next, install cpp extensions for pointnet:
```shell
cd PISA/openpoints/cpp/pointnet2_batch
python setup.py install
```

### Visualization

Install `jupyterlab` and interative widgets:
```shell
pip install trame==2.5 jupyter-server-proxy
```

#### Recommended Test Scene
```
(['scene0474_00'], ['Create a chair on the ground in the corner.'], 19)
```

## FAQ


If the following error occurs:
```
libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

Then update the libstdc++ library in conda:
```shell
conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge gcc=11
```

If the following error occurs, especially when installing `emd_cuda`:
```shell
gxx_linux-64
```
Then please install the following:
```shell
conda install -c conda-forge gxx_linux-64=11.4
```
