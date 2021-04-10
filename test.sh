export PYTHONPATH=$PYTHONPATH:`pwd`
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --num-gpus 4 \
        --config-file configs/fcos/fcos_imprv_R_101_FPN.yaml \
        --eval-only MODEL.WEIGHTS ./pretrained_models/model_final_beta.pth 2>&1 | tee log/test_log.txt
