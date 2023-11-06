#!/bin/bash

# stop when error
set -e

CUR_DIR=$(dirname $0)

cd $CUR_DIR
# Nr3D only
echo "Nr3D only"
python prepare_scannet_data.py -top-scan-dir ../../datasets/scannet/scans -top-save-dir ../../datasets/scannet/PISA --process-only-zero-view True --n-processes 8
# Nr3D + Sr3D
echo "Nr3D + Sr3D"
python prepare_scannet_data.py -top-scan-dir ../../datasets/scannet/scans -top-save-dir ../../datasets/scannet/PISA --process-only-zero-view False --n-processes 8

# create links
echo "Create links..."
cd ../../datasets/PISA
ln -s keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl global_small.pkl
ln -s keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl global.pkl
