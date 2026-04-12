"""Classification components
"""

import torch
import torch.nn as nn


class VGG11Classifier(nn.Module):

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):

        super().__init__()
        from .vgg11 import VGG11Encoder
        from .layers import CustomDropout
        
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Determine bottleneck size (if input is 224x224, bottleneck is 512x7x7)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes)
        )
        self._init_classifier_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x, return_features=False)
        logits = self.classifier(features)
        return logits

    def _init_classifier_weights(self):
        """Initialize classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
