# Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers [BCNet, CVPR 2021]


This is the official pytorch implementation of [BCNet](https://arxiv.org/abs/2103.12340) built on the open-source detectron2.

> [**Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers**](https://arxiv.org/abs/2103.12340)           
> Lei Ke, Yu-Wing Tai, Chi-Keung Tang    
> *CVPR 2021*  

- **Two-stage instance segmentation with state-of-the-art performance.**
- **Image formation as composition of two overlapping layers.**
- **Bilayer decoupling for the occluder and occludee.**
- **Efficacy on both the FCOS and Faster R-CNN detectors.**

**Under construction**. Our code and pretrained model will be fully released in two months.


Visualization of Occluded Objects
-----------------
<table>
    <tr>
        <td><center><img src="demo/fig_vis2_new.png" height="260">
            
Qualitative instance segmentation results of our BCNet, using ResNet-101-FPN and Faster R-CNN detector. The bottom row visualizes squared heatmap of contour and mask predictions by the two GCN layers for the occluder and occludee in the same ROI region specified by the red bounding box, which also makes the final segmentation result of BCNet more explainable than previous methods.
          </center></td>
</tr>
</table>
<table>
    <tr>
          <td><center><img src="demo/fig_vis1_new.png" height="260">
              
Qualitative instance segmentation results of our BCNet, using ResNet-101-FPN and FCOS detector.
          </center></td>
</tr>
</table>

Results on COCO test-dev
------------
(Check Table 8 of the paper for full results, all methods are trained on COCO train2017)
Detector | Backbone  | Method | mAP(mask) | Pretrained Model|
|--------|----------|--------|-----------|-----------|
Faster R-CNN| ResNet-50 FPN | Mask R-CNN | 34.2 | - |
Faster R-CNN| ResNet-50 FPN | MS R-CNN | 35.6 | - |
Faster R-CNN| ResNet-50 FPN | PointRend | 36.3 | - |
Faster R-CNN| ResNet-50 FPN | PANet | 36.6 | - |
**Faster R-CNN**| **ResNet-50 FPN** | **BCNet** | **38.4** | - |
Faster R-CNN| ResNet-101 FPN | Mask R-CNN | 36.1 | - |
Faster R-CNN| ResNet-101 FPN | BMask R-CNN | 37.7 | - |
Faster R-CNN| ResNet-101 FPN | MS R-CNN | 38.3 | - |
**Faster R-CNN**|**ResNet-101 FPN** | **BCNet** | [**39.8**](scores/stdout_frcnn.txt) | TBD |
FCOS| ResNet-101 FPN | SipMask | 37.8 | - |
FCOS| ResNet-101 FPN | BlendMask | 38.4 | - |
FCOS| ResNet-101 FPN | CenterMask| 38.3 | - |
**FCOS**| **ResNet-101 FPN**| **BCNet** | [**39.6**](scores/stdout_fcos.txt)| TBD |


Introduction
-----------------
Segmenting highly-overlapping objects is challenging, because typically no distinction is made between real object contours and occlusion boundaries. Unlike previous two-stage instance segmentation methods, **BCNet** models image formation as composition of two overlapping layers, where the top GCN layer detects the occluding objects (occluder) and the bottom GCN layer infers partially occluded instance (occludee). **The explicit modeling of occlusion relationship with bilayer structure naturally decouples the boundaries of both the occluding and occluded instances, and considers the interaction between them during mask regression.** We validate the efficacy of bilayer decoupling on both one-stage and two-stage object detectors with different backbones and network layer choices. The network of BCNet is as follows:
<center>
<table>
    <tr>
          <td><center><img src="demo/framework_new.png" height="400"></center></td>
    </tr>
</table>
</center>

## Step-by-step Installation
```
conda create -n bcnet python=3.7 -y
source activate bcnet
 
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
 
# FCOS and coco api and visualization dependencies
pip install ninja yacs cython matplotlib tqdm
pip install opencv-python==4.4.0.40
 
export INSTALL_DIR=$PWD
 
# install pycocotools. Please make sure you have installed cython.
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
 
# install BCNet
cd $INSTALL_DIR
git clone https://github.com/lkeab/BCNet.git
cd BCNet/
python3 setup.py build develop
 
unset INSTALL_DIR
```


## Dataset Preparation
Prepare for [coco2017](http://cocodataset.org/#home) dataset following [this instruction](https://github.com/facebookresearch/detectron2/tree/master/datasets). And use our [converted mask annotations](https://hkustconnect-my.sharepoint.com/:u:/g/personal/lkeab_connect_ust_hk/EW2ZVyev7e5Pr1fVfF2nn18BRod82j_jW5Z4ywYd1evq8Q?e=qj0Bbm) to replace original annotation file for bilayer decoupling training.

```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
  ln -s /path_to_coco_dataset/test2017 datasets/coco/test2017
  ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```

Pretrained Models
---------------
TBD
```
  mkdir pretrained_models
  #The pretrained models will be downloaded when running the program.
```

Multi-GPU Training
---------------
TBD

Testing
---------------
TBD

<!---
Install
-----------------
  Check [INSTALL.md](INSTALL.md) for installation instructions.
Prepare Data
----------------
```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
  ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
  ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
```
Pretrained Models
---------------
```
  mkdir pretrained_models
  #The pretrained models will be downloaded when running the program.
```
My training log and pre-trained models can be found here [link](https://1drv.ms/f/s!AntfaTaAXHobhkCKfcPPQQfOfFAB) or [link](https://pan.baidu.com/s/192lRQozksu5XwpU9EO5neg)(pw:xm3f).
Running
----------------
Single GPU Training
```
  python tools/train_net.py --config-file "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1
```
Multi-GPU Training
```
  export NGPUS=8
  python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/e2e_ms_rcnn_R_50_FPN_1x.yaml" 
```
-->

Citations
---------------
If you find BCNet useful in your research, please star this repository and consider citing:
```
@inproceedings{ke2021bcnet,
    author = {Ke, Lei and Tai, Yu-Wing and Tang, Chi-Keung},
    title = {Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers},
    booktitle = {CVPR},
    year = {2021},
}   
```

License
---------------
BCNet is released under the MIT license. See [LICENSE](LICENSE) for additional details.
Thanks to the Third Party Libs
[detectron2](https://github.com/facebookresearch/detectron2)    
