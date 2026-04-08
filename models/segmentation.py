"""Segmentation model
"""

import torch
import torch.nn as nn

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        from .vgg11 import VGG11Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Contracting path feature sizes: Default input (3, 224, 224) 
        # conv1: 64,  conv2: 128, conv3: 256, conv4: 512, conv5: 512
        
        # Expansive path
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self._dec_block(1024, 512, dropout_p) # 512 (conv5 skip) + 512 (up)
        
        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec3 = self._dec_block(1024, 256, dropout_p) # 512 (conv4 skip) + 512 (up)
        
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = self._dec_block(512, 128, dropout_p) # 256 (conv3 skip) + 256 (up)
        
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = self._dec_block(256, 64, dropout_p)  # 128 (conv2 skip) + 128 (up)
        
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec0 = self._dec_block(128, 64, dropout_p)  # 64 (conv1 skip) + 64 (up)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _dec_block(self, in_features: int, out_features: int, dropout_p: float):
        from .layers import CustomDropout
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, features = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, H/32, W/32] from conv5
        
        # Match dimensions appropriately by slicing or padding if needed. 
        # But for well formed multiples of 32 like 224, it handles naturally.
        x = self.up4(bottleneck)
        x = torch.cat([x, features["conv5"]], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, features["conv4"]], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, features["conv3"]], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, features["conv2"]], dim=1)
        x = self.dec1(x)
        
        x = self.up0(x)
        x = torch.cat([x, features["conv1"]], dim=1)
        x = self.dec0(x)
        logits = self.final_conv(x)
        
        return logits
