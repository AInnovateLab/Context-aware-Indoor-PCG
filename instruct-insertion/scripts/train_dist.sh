cd $(dirname $0)
# Possible default args:
export PATH_OF_SCANNET_FILE="/home/hyx/workspace_znk/pisa/datasets/scannet_instruct_global.pkl"
export PATH_OF_REFERIT3D_FILE="/home/hyx/workspace_znk/pisa/datasets/nr3d_generative_20230825.csv"
export PATH_OF_REFERIT3D_SR3D_FILE="/home/hyx/workspace_znk/pisa/datasets/sr3d+_generative_20230918.csv"
export PATH_OF_BERT="bert-base-uncased"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export HTTP_PROXY="http://114.214.236.129:7890"
export HTTPS_PROXY="http://114.214.236.129:7890"

# NOTE: Run `accelerate config` to setup proper config file at first.

accelerate launch train.py \
    --scannet-file $PATH_OF_SCANNET_FILE \
    --referit3D-file $PATH_OF_REFERIT3D_FILE \
    --project-top-dir "../../runs/240402_anorm_b32_2048_fps_800k_sr3d_sim" \
    --project-name "default" \
    --config-file "../configs/config.json" \
    --mentions-target-class-only False \
    --points-per-object 2048 \
    --pretrained-point-e True \
    --axis-norm True \
    --axis-norm-bins 32 \
    --fps True \
    --random-rotation True \
    --gradient-accumulation-steps 1 \
    --point-e-only False \
    --augment-with-sr3d $PATH_OF_REFERIT3D_SR3D_FILE
    # --resume-path /home/hyx/workspace_znk/pisa/runs/240326_anorm_b32_fps_800k_sr3d_sim/default/checkpoints/2024-03-26_22-04-30/ckpt_420000
    
