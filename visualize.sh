python3 setup.py build develop #--no-deps
export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/demo.py --config-file configs/fcos/fcos_imprv_R_101_FPN.yaml \
  --input 'test_img/' \
  --output 'result_img/' \
  --opts MODEL.WEIGHTS ./pretrained_models/model_final_beta.pth

#detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
