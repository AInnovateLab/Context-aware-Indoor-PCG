Thank you for your interest in ScanNet. Please use the following script to download the ScanNet data: http://kaldir.vc.in.tum.de/scannet/download-scannet.py.
Note that by default this script will download the newest (v2) release of the ScanNet data (see the changelog); older versions can still be downloaded by specifying the version (e.g., --v1).

Some useful info:
Scan data is named by scene[spaceid]_[scanid], or scene%04d_%02d, where each space corresponds to a unique location (0-indexed).
Script usage:
- To download the entire ScanNet release (1.3TB): download-scannet.py -o [directory in which to download]
- To download a specific scan (e.g., scene0000_00): download-scannet.py -o [directory in which to download] --id scene0000_00
- To download a specific file type (e.g., *.sens, valid file suffixes listed here): download-scannet.py -o [directory in which to download] --type .sens
- To download the ScanNet v1 task data (inc. trained models): download-scannet.py -o [directory in which to download] --task_data
-  Train/test splits are given in the main ScanNet project repository: https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark
- ScanNet200 preprocessing information: https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/ScanNet200; to download the label map file: download-scannet.py -o [directory in which to download] --label_map

License: ScanNet data is released under the Terms of Use; code is released under the MIT license.
