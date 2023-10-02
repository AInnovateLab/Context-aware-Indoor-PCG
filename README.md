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
```shell
conda create -n instruct-insertion python=3.9
conda activate instruct-insertion
conda install -c conda-forge cudatoolkit-dev=11.7
pip install -r requirements.txt
```

Install chamfer: (TODO, remove this unnecessary dependency)
```shell
cd instruct-insertion/openpoints/cpp/chamfer_dist
python setup.py install
```

Next, install cpp extensions for pointnet:
```shell
cd instruct-insertion/openpoints/cpp/pointnet2_batch
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
If you want to cauculate EMD, please run the following command:
```shell
cd instruct-insertion/openpoints/cpp/emd
python setup.py install
```


If the following error occurs:
```
libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

Then update the libstdc++ library in conda:
```shell
conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge gcc=11
```

If the following error occurs:
```shell
gxx_linux-64
```
Then please install the following:
```shell
conda install -c conda-forge gxx_linux-64=11.4
```
