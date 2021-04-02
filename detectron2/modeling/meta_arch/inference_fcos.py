import torch

import logging
import math
from typing import List
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

def permute_to_N_HW_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, K, H, W) to (N, (HxW), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, K, H, W)
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HW,K)
    return tensor


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward_for_single_image(
            self, locations, box_cls,
            box_regression, centerness,
            image_size):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        '''
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        '''
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        for i in range(len(box_cls)): # different feature levels
            candidate_inds = box_cls[i].sigmoid() > self.pre_nms_thresh
            pre_nms_top_n = candidate_inds.view(1, -1).sum(1)
            pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

            per_box_cls = box_cls[i].sigmoid() * centerness[i].sigmoid()
            per_candidate_inds = candidate_inds #[i]
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] #+ 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[i][per_box_loc]
            per_pre_nms_top_n = pre_nms_top_n #[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n.item(), sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxes_all.append(detections)
            scores_all.append(torch.sqrt(per_box_cls))
            class_idxs_all.append(per_class)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_thresh)

        result = Instances(image_size)
        if not self.training:
            keep = keep[: self.fpn_post_nms_top_n]
            result.pred_boxes = Boxes(boxes_all[keep])
            result.scores = scores_all[keep]
            result.pred_classes = class_idxs_all[keep]
        else:
            keep = keep[: self.fpn_post_nms_top_n] # now 100 for train, change it? Now I use all to train
            result.proposal_boxes = Boxes(boxes_all[keep])
            result.objectness_logits = scores_all[keep]

        return result

    def forward(self, locations, box_cls, box_regression, centerness, batched_inputs, images):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        results = []
        image_num = box_cls[0].shape[0]
        num_classes = 80
        box_cls = [permute_to_N_HW_K(x, num_classes) for x in box_cls]
        box_regression = [permute_to_N_HW_K(x, 4) for x in box_regression]
        centerness = [permute_to_N_HW_K(x, 1) for x in centerness]

        for img_idx in range(image_num):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_regression_per_image = [box_regression_per_level[img_idx] for box_regression_per_level in box_regression]

            centerness_per_image = [centerness_per_level[img_idx] for centerness_per_level in centerness]
            results_per_image = self.forward_for_single_image(
                locations, box_cls_per_image, box_regression_per_image, centerness_per_image, tuple(image_size)
            )
            results.append(results_per_image)

        return results

def make_fcos_postprocessor(config): #, is_train):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMAGE
    bbox_aug_enabled = False #config.TEST.BBOX_AUG.ENABLED

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
