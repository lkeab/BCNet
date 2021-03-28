python3 setup.py build develop #--no-deps
export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159

CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --num-gpus 2 \
	--config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml 2>&1 | tee log/train_log_$ID.txt


