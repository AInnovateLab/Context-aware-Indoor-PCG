#!/bin/bash
export PATH_OF_SCANNET_FILE="datasets/scannet/mvt3dvg/global.pkl"
export PATH_OF_REFERIT3D_FILE="datasets/nr3d/nr3d_generative_20230825_final.csv"
export PATH_OF_REFERIT3D_SR3D_FILE="datasets/sr3d/sr3d+_generative_20230918.csv"
export PATH_OF_BERT="bert-base-uncased"

cd $(dirname $0)

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

python referit3d/scripts/train_referit3d.py \
    -scannet-file $PATH_OF_SCANNET_FILE \
    -referit3D-file $PATH_OF_REFERIT3D_FILE \
    --bert-pretrain-path $PATH_OF_BERT \
    --log-dir logs/MVT_nr3d \
    --n-workers 8 \
    --model 'referIt3DNet_transformer' \
    --unit-sphere-norm True \
    --batch-size 24 \
    --max-train-epochs 30 \
    --encoder-layer-num 3 \
    --decoder-layer-num 4 \
    --decoder-nhead-num 8 \
    --gpu "0" \
    --view_number 4 \
    --rotate_number 4 \
    --label-lang-sup True \
    --augment-with-sr3d $PATH_OF_REFERIT3D_SR3D_FILE
