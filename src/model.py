import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class DGSA(nn.Module):
    """Depth-Guided Spatial Attention"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat):
        # Generate spatial attention map from depth features
        attention_map = self.conv(depth_feat)
        return rgb_feat + rgb_feat * attention_map

class MultiTaskHandModel(nn.Module):
    def __init__(self, num_classes=10, use_depth=True, version=4):
        super().__init__()
        self.use_depth = use_depth
        self.version = version
        
        # RGB branch (using pretrained weights as Encoder)
        self.rgb_backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
        
        if use_depth:
            # Depth branch
            self.depth_backbone = convnext_tiny(weights=None).features
            # Modify the first layer convolution of depth map to accept 1 channel input
            self.depth_backbone[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
            self.fusion = DGSA(768) # ConvNeXt-Tiny last layer dimension is 768

        # --- Task heads ---
        # Classification head
        if self.version == 1:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(768, num_classes)
            )
        else:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(768, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        # Detection head (predict BBox [x1, y1, x2, y2])
        if self.version <= 3:
            self.det_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(768, 4)
            )
        else:
            self.det_reduce_high = nn.Conv2d(768, 384, kernel_size=1)
            self.det_head = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 4, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Sigmoid()
            )
        
        # Segmentation head
        if self.version <= 2:
            self.seg_head = nn.Sequential(
                nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2), # 7->14
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # 14->28
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 28->56
                nn.ReLU(),
                nn.Upsample(scale_factor=4, mode='bilinear'), # 56->224
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.seg_up1 = nn.Sequential(nn.ConvTranspose2d(768, 384, 2, 2), nn.ReLU()) # 7->14
            self.seg_up2 = nn.Sequential(nn.ConvTranspose2d(384, 192, 2, 2), nn.ReLU()) # 14->28
            self.seg_up3 = nn.Sequential(nn.ConvTranspose2d(192, 96, 2, 2), nn.ReLU())  # 28->56
            self.seg_final = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear'), # 56->224
                nn.Conv2d(96, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, rgb, depth=None):
        # --- Extract RGB intermediate layer features (Skip Connections) ---
        # ConvNeXt Tiny structure: [1]=56x56(96), [3]=28x28(192), [5]=14x14(384), [7]=7x7(768)
        s1 = self.rgb_backbone[0:2](rgb)    # 56x56
        s2 = self.rgb_backbone[2:4](s1)     # 28x28
        s3 = self.rgb_backbone[4:6](s2)     # 14x14
        x_rgb = self.rgb_backbone[6:8](s3)  # 7x7
        
        if self.use_depth and depth is not None:
            x_depth = self.depth_backbone(depth)
            # DGSA fusion (performed at the bottom layer 7x7)
            x_fused = self.fusion(x_rgb, x_depth)
        else:
            x_fused = x_rgb
            
        # --- Task outputs ---
        cls_out = self.cls_head(x_fused)
        if self.version <= 3:
            det_out = self.det_head(x_fused)
        else:
            # 7x7 -> 14x14
            high_up = F.interpolate(self.det_reduce_high(x_fused), scale_factor=2, mode='bilinear', align_corners=False)
            # Merge with middle layer s3 (14x14)
            det_feat = high_up + s3
            det_out = self.det_head(det_feat)
        
        if self.version <= 2:
            seg_out = self.seg_head(x_fused)
        else:
            # 7x7 -> 14x14 and merge with s3
            u1 = self.seg_up1(x_fused) + s3
            # 14x14 -> 28x28 and merge with s2
            u2 = self.seg_up2(u1) + s2
            # 28x28 -> 56x56 and merge with s1
            u3 = self.seg_up3(u2) + s1
            # 56x56 -> 224x224
            seg_out = self.seg_final(u3)
        
        return cls_out, det_out, seg_out