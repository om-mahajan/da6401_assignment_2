"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction method: {reduction}")
        self.reduction = reduction

    def _get_corners(self, boxes: torch.Tensor): #-> torch.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
        x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return x1, y1, x2, y2

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        pred_x1, pred_y1, pred_x2, pred_y2 = self._get_corners(pred_boxes)
        target_x1, target_y1, target_x2, target_y2 = self._get_corners(target_boxes)

        # Intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union area
        pred_area = torch.clamp(pred_x2 - pred_x1, min=0) * torch.clamp(pred_y2 - pred_y1, min=0)
        target_area = torch.clamp(target_x2 - target_x1, min=0) * torch.clamp(target_y2 - target_y1, min=0)
        union_area = pred_area + target_area - inter_area

        # Compute IoU
        iou = inter_area / (union_area + self.eps)

        # IoU Loss is 1 - IoU
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss