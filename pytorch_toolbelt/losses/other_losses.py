import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from .functional import tversky_score
from torch import Tensor

__all__ = ["BceLoss", "TverskyLoss", "FocalTverskyLoss"]

class BceLoss(_Loss):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            target.unsqueeze(1),
            pos_weight=self.pos_weight,
        )
        return bce_loss        

class TverskyLoss(_Loss):
    def __init__(
        self,
        alpha=0.7,
        log_loss=False,
        from_logits=True,
        smooth=0,
        eps=1e-7,
    ):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = y_pred.sigmoid()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        scores = tversky_score(
            y_pred, y_true.type(y_pred.dtype), self.alpha, self.smooth, self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores)
        else:
            loss = 1 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = (y_true.sum(dims) > 0).float()
        loss = loss * mask

        return loss.mean()


class FocalTverskyLoss(_Loss):
    def __init__(
        self,
        alpha=0.7,
        gamma=0.75,
        from_logits=True,
        smooth=0,
        eps=1e-7,
    ):
        super(FocalTverskyLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = y_pred.sigmoid()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        scores = tversky_score(
            y_pred, y_true.type(y_pred.dtype), self.alpha, self.smooth, self.eps, dims=dims
        )

        loss = torch.pow((1 - scores), self.gamma)

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = (y_true.sum(dims) > 0).float()
        loss = loss * mask

        return loss.mean()

