cd $(dirname $0)
# Possible default args:
export PATH_OF_SCANNET_FILE="../../datasets/scannet/instruct/global.pkl"
export PATH_OF_REFERIT3D_FILE="../../datasets/nr3d/nr3d_generative_20230825_c4.csv"
export PATH_OF_BERT="bert-base-uncased"

python train.py \
    --scannet-file $PATH_OF_SCANNET_FILE \
    --referit3D-file $PATH_OF_REFERIT3D_FILE \
    --project-top-dir "../../tmp_lyy_saves" \
    --project-name "test1" \
    --config-file "../configs/config.json"
