# PISA: Point Cloud-based Instructed Scene Augmentation

Official implementation of the paper "[PISA: Point Cloud-based Instructed Scene Augmentation](TODO)".

## Clone Repo

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

We use [accelerate](https://huggingface.co/docs/accelerate/index) to speed up the training process. Please read following [instructions](https://huggingface.co/docs/accelerate/basic_tutorials/install#configuring--accelerate) to configure your `accelerate` environment.
```shell
accelerate config
```

Compile modules for EMD-based metrics and PointNet:
```shell
pushd PISA/openpoints/cpp/chamfer_dist; python setup.py install; popd
pushd PISA/openpoints/cpp/emd; python setup.py install; popd
pushd PISA/openpoints/cpp/pointnet2_batch; python setup.py install; popd
```

## Visualization

Install `jupyterlab` and interative widgets:
```shell
pip install jupyterlab
pip install trame==2.5 jupyter-server-proxy
```

Check [Visualization](PISA/visualization/README.md) for details.

## FAQ

If the error about `gxx_linux` occurs, especially when compiling `emd_cuda`, please install the following:
```shell
conda install -c conda-forge gxx_linux-64=11.4
```

If the following error occurs:
```
libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

Then update the `libstdc++` library in conda:
```shell
conda install -c conda-forge libstdcxx-ng
conda install -c conda-forge gcc=11
```
