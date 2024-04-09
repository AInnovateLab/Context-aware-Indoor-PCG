cd $(dirname $0)
# Possible default args:
export PATH_OF_SCANNET_FILE="/root/blueprint/PISA_dev/datasets/scannet_instruct_global.pkl"
export PATH_OF_REFERIT3D_FILE="/root/blueprint/PISA_dev/datasets/nr3d_generative_20230825.csv"
export PATH_OF_BERT="bert-base-uncased"
export CUDA_VISIBLE_DEVICES=0,1,3
export HTTP_PROXY="http://localhost:7890"
export HTTPS_PROXY="http://localhost:7890"

python train.py \
    --scannet-file $PATH_OF_SCANNET_FILE \
    --referit3D-file $PATH_OF_REFERIT3D_FILE \
    --project-top-dir "../../tmp_saves" \
    --project-name "240320" \
    --config-file "../configs/config.json"
