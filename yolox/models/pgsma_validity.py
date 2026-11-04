"""Training-only conversion of geometric letterbox masks to YOLOX anchors.

Inference remains unchanged and ONNX does not receive validity masks.
Strides select the anchor center in source-image pixel coordinates.
"""
import torch


def valid_anchors_from_image_mask(valid_image_mask, features, strides=(8,16,32)):
    if valid_image_mask.ndim != 4 or valid_image_mask.shape[1] != 1:
        raise ValueError('valid_image_mask must be [B,1,H,W]')
    if len(features) != len(strides):
        raise ValueError('features and strides must have same length')
    b, _, h, w = valid_image_mask.shape
    flags = []
    for feature, stride in zip(features,strides):
        stride = int(stride)
        if stride < 1 or h % stride or w % stride:
            raise ValueError(f'Canvas {(h,w)} not stride-aligned for {stride}')
        expected=(b,h//stride,w//stride)
        observed=(int(feature.shape[0]),int(feature.shape[-2]),int(feature.shape[-1]))
        if observed != expected:
            raise ValueError(f'Feature geometry {observed} != mask grid {expected}')
        center_flags = valid_image_mask[:,0,stride//2::stride,stride//2::stride] > 0.5
        flags.append(center_flags.flatten(1))
    return torch.cat(flags,dim=1)
