import torch.nn as nn
from nets.resnet50 import resnet50
from nets.rpn import RegionProposalNetwork
from nets.classifier import Resnet50ROIHead


###### 定义Faster R-CNN主网络 ######


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, mode = "training", feat_stride = 16, anchor_scales = [8, 16, 32], ratios = [0.5, 1, 2]):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride  # 下采样倍数
        
        # 获得特征提取部分及分类头
        self.extractor, classifier = resnet50()
        # 构建RPN网络
        self.rpn = RegionProposalNetwork(1024, 512, ratios = ratios, anchor_scales = anchor_scales, feat_stride = self.feat_stride, mode = mode)
        # 构建分类头
        self.head = Resnet50ROIHead(n_class = num_classes + 1, roi_size = 14, spatial_scale = 1, classifier = classifier)
    
    def forward(self, x, scale = 1.):
        # 输入图片尺寸
        img_size = x.shape[2:]
        # 特征提取网络提取特征
        base_feature = self.extractor.forward(x) # (n, 1024, 38, 38)
        # 获得区域候选框
        _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
        # 输入分类头得到分类结果和回归结果
        roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
        return roi_cls_locs, roi_scores, rois, roi_indices