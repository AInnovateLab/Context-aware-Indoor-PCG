# PISA: Point Cloud-based Instructed Scene Augmentation

Official implementation of the paper "[](TODO)".

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

## Train & Evaluation
Train on Nr3D:
```shell
pushd PISA/scripts; bash train.sh; popd
```

Train on the combination of Sr3D and Nr3D:
```shell
pushd PISA/scripts; bash train_sr3d.sh; popd
```

If you need to turn on evaluation mode, please add `--mode test` to the end of the training script.

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

## Contribution

Install pre-commit-hooks before commits:
```shell
pip install pre-commit
pre-commit install
```
