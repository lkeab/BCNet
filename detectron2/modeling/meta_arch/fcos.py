import logging
import math
import torch
from torch import nn

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.logger import log_first_n

from ..backbone import build_backbone
from .build import META_ARCH_REGISTRY

import torch.nn.functional as F

from .inference_fcos import make_fcos_postprocessor
from .loss_fcos import make_fcos_loss_evaluator

from detectron2.layers import Scale
#from detectron2.layers import DFConv2d
from detectron2.modeling.roi_heads.mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss
from ..poolers import ROIPooler
from detectron2.layers import ShapeSpec
from ..sampling import subsample_labels
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..matcher import Matcher
from detectron2.utils.events import get_event_storage
from ..postprocessing import detector_postprocess

__all__ = ["FCOS"]

def select_foreground_proposals(train_part, proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    #voc_ls = [0, 1, 2, 3, 4, 5, 6, 8, 13, 14, 15, 16, 17, 18, 19, 39, 56, 58, 60, 62]
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        if train_part == 'voc':
            fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) & ((gt_classes == 0) | (gt_classes == 1) | (gt_classes == 2) | (gt_classes == 3) | (gt_classes == 4) | (gt_classes == 5) | (gt_classes == 6) | (gt_classes == 8) | (gt_classes == 13) | (gt_classes == 14) | (gt_classes == 15) | (gt_classes == 16) | (gt_classes == 17) | (gt_classes == 18) | (gt_classes == 19) | (gt_classes == 39) | (gt_classes == 56) | (gt_classes == 58) | (gt_classes == 60) | (gt_classes == 62)) # train on voc
        elif train_part == 'non_voc':
            fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) & (gt_classes != 0) & (gt_classes != 1) & (gt_classes != 2) & (gt_classes != 3) & (gt_classes != 4) & (gt_classes != 5) & (gt_classes != 6) & (gt_classes != 8) & (gt_classes != 13) & (gt_classes != 14) & (gt_classes != 15) & (gt_classes != 16) & (gt_classes != 17) & (gt_classes != 18) & (gt_classes != 19) & (gt_classes != 39) & (gt_classes != 56) & (gt_classes != 58) & (gt_classes != 60) & (gt_classes != 62) # train on non voc
        elif train_part == 'all':
            fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) # train on all categories
        else:
            assert False
        #for cnt, i in enumerate(gt_classes):
        #    if i not in voc_ls: # train on voc
        #        fg_selection_mask[cnt] = False
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks

@META_ARCH_REGISTRY.register()
class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.in_features              = cfg.MODEL.FCOS.IN_FEATURES
    
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        #box_selector_train = make_fcos_postprocessor(cfg, is_train=True)
        box_selector = make_fcos_postprocessor(cfg) #, is_train=False)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        #self.box_selector_train = box_selector_train
        self.box_selector = box_selector
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.head = FCOSHead(cfg, feature_shapes[0].channels)

        # train on voc or non-voc or all
        self.train_part = cfg.MODEL.FCOS.TRAIN_PART

        self.feature_strides          = {k: v.stride for k, v in backbone_shape.items()}

        # mask head
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on
        
        in_channels = feature_shapes[0].channels

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)



    def forward(self, batched_inputs, c_iter, max_iter):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features_list = [features[f] for f in self.in_features]
        box_cls, box_regression, centerness = self.head(features_list)
        locations = self.compute_locations(features_list)
 
        if self.training:
            return self._forward_train(
                features_list,
                locations, box_cls, 
                box_regression, 
                centerness, gt_instances, batched_inputs, images, c_iter, max_iter
            )
        else:
            #assert False
            return self._forward_test(
                features_list,
                locations, box_cls, box_regression, 
                centerness, batched_inputs, images
            )

    def _forward_train(self, features_list, locations, box_cls, box_regression, centerness, gt_instances, batched_inputs, images, c_iter, max_iter):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, gt_instances
        )
        #print('fcos.py 193 gt_instances:', gt_instances)
        proposals = self.box_selector(
            locations, box_cls, box_regression, 
            centerness, batched_inputs, images
        )
        #print('fcos.py 198 proposals:', proposals)
        proposals = self.label_and_sample_proposals(proposals, gt_instances)
        #print('fcos.py 200 proposals after label_sample:', proposals)
        del gt_instances
        loss_mask, loss_mask_bo, loss_boundary, loss_boundary_bo = self._forward_mask(features_list, proposals)
        loss_rate = (float(c_iter)/max_iter) * 1.0
        #print('loss rate:', loss_rate)
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg * 1.25,
            "loss_centerness": loss_centerness,
            "loss_mask": loss_mask,
            "loss_mask_bo": loss_mask_bo * 0.25,
            "loss_boundary_bo": loss_boundary_bo * 0.5,
            "loss_boundary": loss_boundary * 0.5
        }
        return losses
    
    def _forward_test(self, features, locations, box_cls, box_regression, centerness, batched_inputs, images):
        instances = self.box_selector(
            locations, box_cls, box_regression, 
            centerness, batched_inputs, images
        )
        
        # mask
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        instances = self._forward_mask(features, instances)
        
        return self._postprocess(instances, batched_inputs, images.image_sizes)

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(self.train_part, instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            #if len(mask_features) == 0:
            #    print('No X.............................')
            #    return 0.0, 0.0, 0.0
            #else:
            mask_logits, boundary, bo_masks, bo_bound = self.mask_head(mask_features)
            return mask_rcnn_loss(mask_logits, boundary, proposals, bo_masks, bo_bound)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits, boundary, bo_masks, bo_bound = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, bo_masks, boundary, bo_bound, instances)
            return instances

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if True: #self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        #num_fg_samples = []
        #num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            #num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            #num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            #print(num_bg_samples, num_fg_samples)
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        #storage = get_event_storage()
        #storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        #storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.nonzero((gt_classes != -1) & (gt_classes != self.num_classes)).squeeze(1)
        '''
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        '''
        return sampled_idxs, gt_classes[sampled_idxs]

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class FCOSHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            #if self.use_dcn_in_tower and \
            #        i == cfg.MODEL.FCOS.NUM_CONVS - 1:
            #    conv_func = DFConv2d
            #else:
            conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness

