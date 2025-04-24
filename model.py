from torchvision.models.detection.mask_rcnn import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.ops import nms
from torchvision.transforms.v2 import functional as F
import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils
import torch
import os
import json
import numpy as np
import skimage.io as sio
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import cv2
from pycocotools import mask as mask_utils
import random
from tqdm import tqdm
from torchvision.ops import box_iou
import os
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import DeformConv2d


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResNetWithCBAM(ResNet):
    def __init__(self, *args, cbam_ratio=16, cbam_kernel=7, **kwargs):
        super().__init__(*args, **kwargs)
        state_dict = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        ).state_dict()
        self.load_state_dict(state_dict, strict=False)

        for name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self, name)
            out_channels = (
                layer[-1].conv3.out_channels
                if hasattr(layer[-1], "conv3")
                else layer[-1].conv2.out_channels
            )
            cbam = CBAM(out_channels, ratio=cbam_ratio, kernel_size=cbam_kernel)
            layer.add_module("cbam", cbam)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def get_model(num_classes, hidden_layer=256, trainable_backbone_layers=3):
    backbone = ResNetWithCBAM(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        cbam_ratio=16,
        cbam_kernel=7,
        norm_layer=torch.nn.BatchNorm2d,
    )
    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
    )
    model = torchvision.models.detection.MaskRCNN(
        backbone_with_fpn,
        num_classes=num_classes,
    )
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features_box, num_classes
        )
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model
