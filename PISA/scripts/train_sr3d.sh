#!/bin/bash
# Script to train the model on both Nr3D-SA and Sr3D-SA.

cd $(dirname $0)
# Possible default args:
export PATH_OF_SCANNET_FILE="../../datasets/scannet/PISA/global.pkl"
export PATH_OF_REFERIT3D_FILE="../../datasets/nr3d/nr3d_sa.csv"
export PATH_OF_REFERIT3D_SR3D_FILE="../../datasets/sr3d/sr3d+_sa.csv"
export PATH_OF_BERT="bert-base-uncased"
export ACCELERATE_LOG_LEVEL="INFO"

# NOTE: Change the following two lines to your own path.
# The project-related files will be saved in $PATH_TO_TOP_DIR/$PROJECT_NAME
PATH_TO_TOP_DIR="TODO"
PROJECT_NAME="fps_qpp32_rr4_sr3d"

accelerate launch train.py \
    --scannet-file $PATH_OF_SCANNET_FILE \
    --referit3D-file $PATH_OF_REFERIT3D_FILE \
    --project-top-dir "$PATH_TO_TOP_DIR" \
    --project-name "$PROJECT_NAME" \
    --config-file "../configs/config_sr3d.json" \
    --mentions-target-class-only False \
    --points-per-object 1024 \
    --pretrained-point-e True \
    --axis-norm True \
    --axis-norm-bins 32 \
    --fps True \
    --random-rotation True \
    --gradient-accumulation-steps 1 \
    --point-e-only False \
    --augment-with-sr3d $PATH_OF_REFERIT3D_SR3D_FILE
