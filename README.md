# PISA: Point Cloud based Instructed Scene Augmentation

## Install Openpoints

Pull the repo with submodules:
```shell
git clone --recurse-submodules <Our-Git-Repo>
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

Check [Dataset](datasets/README.md) for more details.

## Setup
Create environment:
```shell
conda create -n pisa python=3.9
conda activate pisa
conda install -c conda-forge cudatoolkit-dev=11.7
pip install -r requirements.txt
```

We use [accelerate](https://huggingface.co/docs/accelerate/index) to speed up the training process. Please install following [instructions](https://huggingface.co/docs/accelerate/basic_tutorials/install)

If you want to cauculate MMD and 1-NNA, please run the following command:
```shell
pushd PISA/openpoints/cpp/emd
python setup.py install
popd
```

Next, install cpp extensions for pointnet:
```shell
pushd PISA/openpoints/cpp/pointnet2_batch
python setup.py install
popd
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
