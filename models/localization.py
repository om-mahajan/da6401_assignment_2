"""Localization modules
"""

import torch
import torch.nn as nn

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        from .vgg11 import VGG11Encoder
        from .layers import CustomDropout
        
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid()  # Assuming we scale outputs later
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        features = self.encoder(x, return_features=False)
        
        # We output a normalized [0, 1] relative to image size.
        # Multiply by the image dimension height and width
        norm_boxes = self.regression_head(features)
        
        # Multiply x_c, y_c, w, h by the dimensions respectively
        B, C, H, W = x.shape
        scale_tensor = torch.tensor([W, H, W, H], dtype=x.dtype, device=x.device)
        boxes = norm_boxes * scale_tensor
        return boxes
