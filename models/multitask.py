"""Unified multi-task model
"""

import torch
import torch.nn as nn

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            dropout_p: Custom dropout rate
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weig          hts.
            unet_path: Path to trained unet weights.
        """
        import gdown
        
        gdown.download(id="1qPdQNpPJ6-1adyUyJ-8oYZwj4K3QBXxi", output=classifier_path, quiet=False)
        gdown.download(id="1Hvl63xqdC5g3Xb1GKgjtvHH-jT7YjZro", output=localizer_path, quiet=False)
        #gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)
        super().__init__()
        # Wait until downloaded or loaded. Just initialize heads.
        from .vgg11 import VGG11Encoder
        from .classification import VGG11Classifier
        from .localization import VGG11Localizer
        from .segmentation import VGG11UNet
        
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # Load weights
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels, dropout_p=dropout_p)
        
        state_dict = torch.load(classifier_path, map_location="cpu")
        classifier.load_state_dict(state_dict)
            
        self.classification_head = classifier.classifier
        self.encoder = classifier.encoder # use shared encoder from classifier

        localizer = VGG11Localizer(in_channels=in_channels, dropout_p=dropout_p)
        loc_state_dict = torch.load(localizer_path, map_location="cpu")
        localizer.load_state_dict(loc_state_dict)
        self.regression_head = localizer.regression_head
        
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, dropout_p=dropout_p)
        #unet_state_dict = torch.load(seg_path, map_location="cpu")
        #unet.load_state_dict(unet_state_dict)
        self.up4 = unet.up4
        self.dec4 = unet.dec4
        self.up3 = unet.up3
        self.dec3 = unet.dec3
        self.up2 = unet.up2
        self.dec2 = unet.dec2
        self.up1 = unet.up1
        self.dec1 = unet.dec1
        self.up0 = unet.up0
        self.dec0 = unet.dec0
        self.final_conv = unet.final_conv

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bottleneck, features = self.encoder(x, return_features=True)
        
        # Classification
        class_logits = self.classification_head(bottleneck)
        
        # Localization
        norm_boxes = self.regression_head(bottleneck)
        B, C, H, W = x.shape
        scale_tensor = torch.tensor([W, H, W, H], dtype=x.dtype, device=x.device)
        bbox = norm_boxes * scale_tensor
        
        # Segmentation 
        s = self.up4(bottleneck)
        s = torch.cat([s, features["conv5"]], dim=1)
        s = self.dec4(s)
        
        s = self.up3(s)
        s = torch.cat([s, features["conv4"]], dim=1)
        s = self.dec3(s)
        
        s = self.up2(s)
        s = torch.cat([s, features["conv3"]], dim=1)
        s = self.dec2(s)
        
        s = self.up1(s)
        s = torch.cat([s, features["conv2"]], dim=1)
        s = self.dec1(s)
        
        s = self.up0(s)
        s = torch.cat([s, features["conv1"]], dim=1)
        s = self.dec0(s)
        seg_logits = self.final_conv(s)
        
        return {
            'classification': class_logits,
            'localization': bbox,
            'segmentation': seg_logits
        }
