import torch
import numpy as np
import torch.nn as nn
from torchvision.ops import nms
from torch.nn import functional as F
from utils.utils_anchors import generate_anchor_base, _enumerate_shifted_anchor
from utils.utils_bbox import loc2bbox


###### 生成区域建议网络 ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ProposalCreator():
    def __init__(self, mode, nms_iou = 0.7, n_train_pre_nms = 12000, n_train_post_nms = 600, n_test_pre_nms = 3000, n_test_post_nms = 300, min_size = 16, device = device):
        self.mode = mode  # 设置训练or测试模式
        self.nms_iou = nms_iou  # 执行nms时的阈值
        self.n_train_pre_nms = n_train_pre_nms  # 训练时保留的建议框数量
        self.n_train_post_nms = n_train_post_nms  # nms后保留的建议框数量
        self.n_test_pre_nms = n_test_pre_nms  # 测试时保留的建议框数量
        self.n_test_post_nms = n_test_post_nms  # nms后保留的建议框数量
        self.min_size = min_size
        self.device = device  # 代码运行环境
    
    def __call__(self, loc, score, anchor, img_size, scale = 1.):
        # 判断处于训练模式还是测试模式
        if self.mode == 'training':
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        
        anchor = torch.from_numpy(anchor)  # 将anchor转化为tensor
        anchor = anchor.to(self.device)

        # 将RPN网络预测结果转化为经过调整的候选框 (n * w * 9, 4)
        roi = loc2bbox(anchor, loc)

        # 防止区域候选框超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])  # 限制x坐标
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])  # 限制y坐标
        
        # 候选框的最小值不可以小于16， 否则下采样后不足一个像素
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        
        # 保留剩余候选框
        roi = roi[keep, :]
        score = score[keep]

        # 根据得分进行排序
        order = torch.argsort(score, descending = True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]  
        roi = roi[order, :] # 取前n_pre_nms个候选框
        score = score[order]

        # 进行非极大值抑制
        keep = nms(roi, score, self.nms_iou)
        keep = keep[:n_post_nms]  # 取前n_post_nms个候选框
        roi = roi[keep]
        return roi

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels = 512, mid_channels = 512, ratios = [0.5, 1, 2], anchor_scales = [8, 16, 32], feat_stride = 16, mode = 'training'):
        super(RegionProposalNetwork, self).__init__()

        # 生成anchor 形状为(9, 4)
        self.anchor_base = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)
        n_anchor = self.anchor_base.shape[0]

        # 利用3 * 3卷积融合上下文信息
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # 产生n_anchor * 2维向量，指示是否为前景
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        # 产生n_anchor * 4维向量, 调整边界框尺寸
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        # 特征点间距步长
        self.feat_stride = feat_stride

        # 对建议框解码并进行非极大值抑制
        self.proposal_layer = ProposalCreator(mode)
    
    def forward(self, x, img_size, scale = 1.):
        n, _, h, w = x.shape  # 输入特征的形状

        x = self.conv1(x)  # 3 * 3卷积融合上下文信息
        x = F.relu(x)

        rpn_locs = self.loc(x)  # 获取调整参数 （n, 4 * 9, h, w）
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # (n, h * w * 9, 4)

        rpn_scores = self.score(x)  # 获取前景背景概率 (n, 2 * 9, h, w)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)  # (n, h * w * 9, 2)
        
        # 使用softmax计算概率
        rpn_softmax_scores = F.softmax(rpn_scores, dim = -1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()  # 第一维为前景概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # (n, h * w * 9)

        # 生成anchor框
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)

        rois = list()  # 用于堆叠batch内的每个样本
        roi_indices = list()
        
        # 对batch内每一个样本进行处理
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            batch_index = i * torch.ones((len(roi), ))  # (n_post_nms)
            rois.append(roi)  # (n_post_nms, 4)
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim = 0)  # (n_post_nms * n, 4)
        roi_indices = torch.cat(roi_indices, dim = 0)  # (n_post_nms * n, )

        return rpn_locs, rpn_scores, rois, roi_indices, anchor