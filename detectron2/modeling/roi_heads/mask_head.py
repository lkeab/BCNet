import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.layers import interpolate, get_instances_contour_interior
from pytorch_toolbelt import losses as L

from pytorch_toolbelt.modules import AddCoords

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_rcnn_loss(pred_mask_logits, pred_boundary_logits, instances, pred_mask_bo_logits, pred_boundary_logits_bo):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    gt_bo_masks = []
    gt_boundary_bo = []
    gt_boundary = []

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        #print('mask_head.py L59 instances_per_image.gt_masks:', instances_per_image.gt_masks)
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

        boundary_ls = []
        for mask in gt_masks_per_image:
            mask_b = mask.data.cpu().numpy()
            boundary, inside_mask, weight = get_instances_contour_interior(mask_b)
            boundary = torch.from_numpy(boundary).to(device=mask.device).unsqueeze(0)

            boundary_ls.append(boundary)

        gt_boundary.append(cat(boundary_ls, dim=0))

       
        gt_bo_masks_per_image = instances_per_image.gt_bo_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_bo_masks.append(gt_bo_masks_per_image)

        boundary_ls_bo = []
        for mask_bo in gt_bo_masks_per_image:
            mask_b_bo = mask_bo.data.cpu().numpy()
            boundary_bo, inside_mask_bo, weight_bo = get_instances_contour_interior(mask_b_bo)
            boundary_bo = torch.from_numpy(boundary_bo).to(device=mask_bo.device).unsqueeze(0)

            boundary_ls_bo.append(boundary_bo)

        gt_boundary_bo.append(cat(boundary_ls_bo, dim=0))


    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)
    gt_bo_masks = cat(gt_bo_masks, dim=0)

    gt_boundary_bo = cat(gt_boundary_bo, dim=0)
    gt_boundary = cat(gt_boundary, dim=0)
    
    if cls_agnostic_mask:
        pred_mask_logits_gt = pred_mask_logits[:, 0]
        pred_bo_mask_logits = pred_mask_bo_logits[:, 0]
        pred_boundary_logits_bo = pred_boundary_logits_bo[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        gt_masks_bool = gt_masks > 0.5

    mask_incorrect = (pred_mask_logits_gt > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    indexs2 = torch.nonzero(torch.sum(gt_bo_masks.to(dtype=torch.float32),(1,2)))

    new_gt_bo_masks1 = gt_bo_masks[indexs2,:,:].squeeze()
    new_gt_bo_masks2 = gt_bo_masks[:indexs2.shape[0]]
    if new_gt_bo_masks1.shape != new_gt_bo_masks2.shape:
        new_gt_bo_masks1 = new_gt_bo_masks1.unsqueeze(0)

    new_gt_bo_masks = torch.cat((new_gt_bo_masks1, new_gt_bo_masks2),0)
    
    pred_bo_mask_logits1 = pred_bo_mask_logits[indexs2,:,:].squeeze()
    pred_bo_mask_logits2 = pred_bo_mask_logits[:indexs2.shape[0]]
    if pred_bo_mask_logits1.shape != pred_bo_mask_logits2.shape:
        pred_bo_mask_logits1 = pred_bo_mask_logits1.unsqueeze(0)

    new_pred_bo_mask_logits = torch.cat((pred_bo_mask_logits1, pred_bo_mask_logits2),0)
 
    new_gt_bo_bounds1 = gt_boundary_bo[indexs2,:,:].squeeze()
    new_gt_bo_bounds2 = gt_boundary_bo[:indexs2.shape[0]]
    if new_gt_bo_bounds1.shape != new_gt_bo_bounds2.shape:
        new_gt_bo_bounds1 = new_gt_bo_bounds1.unsqueeze(0)

    new_gt_bo_bounds = torch.cat((new_gt_bo_bounds1, new_gt_bo_bounds2),0)
    
    pred_bo_bounds_logits1 = pred_boundary_logits_bo[indexs2,:,:].squeeze()
    pred_bo_bounds_logits2 = pred_boundary_logits_bo[:indexs2.shape[0]]
    if pred_bo_bounds_logits1.shape != pred_bo_bounds_logits2.shape:
        pred_bo_bounds_logits1 = pred_bo_bounds_logits1.unsqueeze(0)

    new_pred_bo_bounds_logits = torch.cat((pred_bo_bounds_logits1, pred_bo_bounds_logits2),0)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits_gt, gt_masks.to(dtype=torch.float32), reduction="mean"
    )

    bound_loss = L.JointLoss(L.BceLoss(), L.BceLoss())(
        pred_boundary_logits.unsqueeze(1), gt_boundary.to(dtype=torch.float32))

    if new_gt_bo_masks.shape[0] > 0: 
        bo_mask_loss = F.binary_cross_entropy_with_logits(
            new_pred_bo_mask_logits, new_gt_bo_masks.to(dtype=torch.float32), reduction="mean"
        )
    else:
        bo_mask_loss = torch.tensor(0.0).cuda(mask_loss.get_device())

    if new_gt_bo_bounds.shape[0] > 0: 
        bo_bound_loss = L.JointLoss(L.BceLoss(), L.BceLoss())(
            new_pred_bo_bounds_logits.unsqueeze(1), new_gt_bo_bounds.to(dtype=torch.float32))
    else:
        bo_bound_loss = torch.tensor(0.0).cuda(mask_loss.get_device())


    return mask_loss, bo_mask_loss, bound_loss, bo_bound_loss



def mask_rcnn_inference(pred_mask_logits, bo_mask_logits, bound_logits, bo_bound_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    #pred_mask_logits = pred_mask_logits[:,0:1]
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
        bound_probs_pred = bound_logits.sigmoid()
        bo_mask_probs_pred = bo_mask_logits.sigmoid()
        bo_bound_probs_pred = bo_bound_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_mask_probs_pred = bo_mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_bound_probs_pred = bo_bound_probs_pred.split(num_boxes_per_image, dim=0)
    bound_probs_pred = bound_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)
        instances.raw_masks = prob

    for bo_prob, instances in zip(bo_mask_probs_pred, pred_instances):
        instances.pred_masks_bo = bo_prob  # (1, Hmask, Wmask)

    for bo_bound_prob, instances in zip(bo_bound_probs_pred, pred_instances):
        instances.pred_bounds_bo = bo_bound_prob  # (1, Hmask, Wmask)

    for bound_prob, instances in zip(bound_probs_pred, pred_instances):
        instances.pred_bounds = bound_prob  # (1, Hmask, Wmask)


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.boundary_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.boundary_deconv_bo = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.boundary_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.bo_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.query_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform_bound_bo = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.query_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform_bound = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)


        self.scale = 1.0 / (input_channels ** 0.5)
        self.blocker_bound_bo = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized
        self.blocker_bound = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.predictor_bo = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.boundary_predictor_bo = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.boundary_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)


        for layer in self.conv_norm_relus + self.boundary_conv_norm_relus + [self.deconv, self.bo_deconv, self.boundary_deconv, self.boundary_deconv_bo, self.query_transform_bound_bo, self.key_transform_bound_bo, self.value_transform_bound_bo, self.output_transform_bound_bo, self.query_transform_bound, self.key_transform_bound, self.value_transform_bound, self.output_transform_bound]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

        nn.init.normal_(self.predictor_bo.weight, std=0.001)
        if self.predictor_bo.bias is not None:
            nn.init.constant_(self.predictor_bo.bias, 0)

        nn.init.normal_(self.boundary_predictor_bo.weight, std=0.001)
        if self.boundary_predictor_bo.bias is not None:
            nn.init.constant_(self.boundary_predictor_bo.bias, 0)

        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)


    def forward(self, x):
        B, C, H, W = x.size()
        x_ori = x.clone()

        for cnt, layer in enumerate(self.boundary_conv_norm_relus):
            x = layer(x)

            if cnt == 1 and len(x) != 0:
                # x: B,C,H,W
                # x_query: B,C,HW
                #x_input = AddCoords()(x)
                x_query_bound_bo = self.query_transform_bound_bo(x).view(B, C, -1)
                # x_query: B,HW,C
                x_query_bound_bo = torch.transpose(x_query_bound_bo, 1, 2)
                # x_key: B,C,HW
                x_key_bound_bo = self.key_transform_bound_bo(x).view(B, C, -1)
                # x_value: B,C,HW
                x_value_bound_bo = self.value_transform_bound_bo(x).view(B, C, -1)
                # x_value: B,HW,C
                x_value_bound_bo = torch.transpose(x_value_bound_bo, 1, 2)
                # W = Q^T K: B,HW,HW
                x_w_bound_bo = torch.matmul(x_query_bound_bo, x_key_bound_bo) * self.scale
                x_w_bound_bo = F.softmax(x_w_bound_bo, dim=-1)
                # x_relation = WV: B,HW,C
                x_relation_bound_bo = torch.matmul(x_w_bound_bo, x_value_bound_bo)
                # x_relation = B,C,HW
                x_relation_bound_bo = torch.transpose(x_relation_bound_bo, 1, 2)
                # x_relation = B,C,H,W
                x_relation_bound_bo = x_relation_bound_bo.view(B,C,H,W)

                x_relation_bound_bo = self.output_transform_bound_bo(x_relation_bound_bo)
                x_relation_bound_bo = self.blocker_bound_bo(x_relation_bound_bo)

                x = x + x_relation_bound_bo

        x_bound_bo = x.clone()

        x_bo = x.clone()
        
        x = x_ori + x

        for cnt, layer in enumerate(self.conv_norm_relus):
            x = layer(x)
            if cnt == 1 and len(x) != 0:
                # x: B,C,H,W
                # x_query: B,C,HW
                #x_input = AddCoords()(x)
                x_query_bound = self.query_transform_bound(x).view(B, C, -1)
                # x_query: B,HW,C
                x_query_bound = torch.transpose(x_query_bound, 1, 2)
                # x_key: B,C,HW
                x_key_bound = self.key_transform_bound(x).view(B, C, -1)
                # x_value: B,C,HW
                x_value_bound = self.value_transform_bound(x).view(B, C, -1)
                # x_value: B,HW,C
                x_value_bound = torch.transpose(x_value_bound, 1, 2)
                # W = Q^T K: B,HW,HW
                x_w_bound = torch.matmul(x_query_bound, x_key_bound) * self.scale
                x_w_bound = F.softmax(x_w_bound, dim=-1)
                # x_relation = WV: B,HW,C
                x_relation_bound = torch.matmul(x_w_bound, x_value_bound)
                # x_relation = B,C,HW
                x_relation_bound = torch.transpose(x_relation_bound, 1, 2)
                # x_relation = B,C,H,W
                x_relation_bound = x_relation_bound.view(B,C,H,W)

                x_relation_bound = self.output_transform_bound(x_relation_bound)
                x_relation_bound = self.blocker_bound(x_relation_bound)

                x = x + x_relation_bound

        x_bound = x.clone()

        x = F.relu(self.deconv(x))
        mask = self.predictor(x) 

        x_bo = F.relu(self.bo_deconv(x_bo))
        mask_bo = self.predictor_bo(x_bo) 

        x_bound_bo = F.relu(self.boundary_deconv_bo(x_bound_bo))
        boundary_bo = self.boundary_predictor_bo(x_bound_bo) 

        x_bound = F.relu(self.boundary_deconv(x_bound))
        boundary = self.boundary_predictor(x_bound) 

        return mask, boundary, mask_bo, boundary_bo


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
