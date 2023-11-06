# Visualization

## Metrics

Run `randomly_sample_test.py` to sample generated objects and save to .pkl files:
```shell
bash run_sample.sh
```

Then run `metrics4pkl.py` to compute metrics:
```shell
python metrics4pkl.py -m acc jsd mmd-emd cov-emd 1-nna-emd -i PATH_TO_TOP_DIR/fps_qpp32_rr4_sr3d/objs.pkl
```

## Interative Jupyter

If you are headless server, use `xvfb` to setup virtual display:
```shell
source headless_display.sh
```

Then run the jupyter server:
```shell
jupyter-lab
```

## FAQ
If the following error occurs:
```
libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```
Or simply get errors like when rendering meshes:
```
vtkXOpenGLRenderWindow.:651    ERR| vtkXOpenGLRenderWindow (0x7f563ac4bc40): Cannot create GLX context.  Aborting.
```

Then update the `libstdc++` library in conda:
```shell
conda install -c conda-forge libstdcxx-ng gcc=11
```
