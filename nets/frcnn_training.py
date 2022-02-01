import torch
import numpy as np
from torch import nn
from collections import namedtuple
from torch.nn import functional as F


###### 定义损失函数 ######


# 计算两个框之间的IOU
def bbox_iou(bbox_a, bbox_b):
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # 重合区域的左上角
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])  # 重合区域的右下角

    area_i = np.prod(br - tl, axis = 2) * (tl < br).all(axis = 2)  # 计算重合区域面积
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis = 1)  # 计算框A面积
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis = 1)  # 计算框B面积
    return area_i / (area_a[:, None] + area_b - area_i)

# 将框间差距转化为拟合参数
def bbox2loc(src_bbox, dst_bbox):
    # 计算参考框的相关参数
    width = src_bbox[:, 2] - src_bbox[:, 0]  # 宽度
    height = src_bbox[:, 3] - src_bbox[:, 1]  # 高度
    ctr_x = src_bbox[:, 0] + 0.5 * width  # 中点x坐标
    ctr_y = src_bbox[:, 1] + 0.5 * height  # 中点y坐标

    # 计算真实框的相关参数
    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]  # 宽度
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]  # 高度
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width  # 中点x坐标
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height  # 中点y坐标
    
    # 将参考框限制在一定范围内
    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width  # x坐标偏移
    dy = (base_ctr_y - ctr_y) / height  # y坐标偏移
    dw = np.log(base_width / width)  # 宽度偏移
    dh = np.log(base_height / height)  # 高度偏移

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc

# 生成训练过程中的正负样本
class AnchorTargetCreator(object):
    def __init__(self, n_sample = 256, pos_iou_thresh = 0.7, neg_iou_thresh = 0.3, pos_ratio = 0.5):
        self.n_sample = n_sample  # 样本数量
        self.pos_iou_thresh = pos_iou_thresh  # 正样本IOU阈值
        self.neg_iou_thresh = neg_iou_thresh  # 负样本IOU阈值
        self.pos_ratio = pos_ratio
    def __call__(self, bbox, anchor):
        # 获取每个anchor对应的最大GT框，及正负样本选择后的结果
        argmax_ious, label = self._create_label(anchor, bbox)
        if (label > 0).any():
            # 将真实框转化为拟合参数
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, label
        else:
            return np.zeros_like(anchor), label
    
    def _calc_ious(self, anchor, bbox):
        ious = bbox_iou(anchor, bbox)  # 计算所有anchor框与所有GT框的IOU值 (num_anchors, num_gt)

        if len(bbox) == 0:
            return np.zeros(len(anchor), np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        
        # 获取每一个anchor框对应的IOU最大的GT框
        argmax_ious = ious.argmax(axis = 1)  # (num_anchors, )
        # 获取每一个anchor框对应IOU最大GT框的IOU值
        max_ious = np.max(ious, axis = 1)  # (num_anchors, )

        # 获取每一个GT框对应的IOU最大的anchor框
        gt_argmax_ious = ious.argmax(axis = 0)  # (num_gt, )
        
        for i in range(len(gt_argmax_ious)):
            argmax_ious[gt_argmax_ious[i]] = i
        return argmax_ious, max_ious, gt_argmax_ious
    def _create_label(self, anchor, bbox):
        # 1是正样本， 0是负样本， -1忽略
        # 初始化的时候全部设置为-1
        label = np.empty((len(anchor), ), dtype = np.int32)  # (n_anchors, )
        label.fill(-1)

        # argmax_ious为每个anchor框对应的IOU最大GT框的序号 （num_anchors，）
        # max_iou为每个anchor框对应的最大IOU值 (num_anchors, )
        # gt_argmax_ious为每一个真实框对应的anchor框的序号 （num_gt, )
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        # 若小于负样本阈值，设置为负样本
        # 若大于负样本阈值，设置为正样本
        # 每个GT框对应的最大anchor设置为正样本
        label[max_ious < self.neg_iou_thresh] = 0
        label[max_ious > self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            label[gt_argmax_ious] = 1
        
        # 将正样本数量限制在128内
        n_pos = int(self.pos_ratio * self.n_sample)  # 正样本数量
        pos_index = np.where(label == 1)[0]  # 正样本索引
        if len(pos_index) > n_pos:
            # 随机选取128个忽略
            disable_index = np.random.choice(pos_index, size = (len(pos_index) - n_pos), replace = False)
            label[disable_index] = -1
        
        # 平衡正负样本，总数为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]  # 负样本索引
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size = (len(neg_index) - n_neg), replace = False)
            label[disable_index] = -1
        
        return argmax_ious, label


class ProposalTargetCreator(object):
    def __init__(self, n_sample = 128, pos_ratio = 0.5, pos_iou_thresh = 0.5, neg_iou_thresh_high = 0.5, neg_iou_thresh_low = 0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)  # 每张图像的正样本数
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low
    def __call__(self, roi, bbox, label, loc_normalize_std = (0.1, 0.1 , 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis = 0)

        # 计算建议框与真实框的重合程度
        iou = bbox_iou(roi, bbox)

        if len(bbox) == 0:
            gt_assignment = np.zeros(len(roi), np.int32)
            max_iou = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            # 每一个anchor框对应的IOU最大的GT框
            gt_assignment = iou.argmax(axis = 1)  # (num_roi, )
            # 每一个anchor框对应的最大IOU值
            max_iou = iou.max(axis = 1)  # (num_roi, )
            # 由于背景，类别+1
            gt_roi_label = label[gt_assignment] + 1

        # 将建议框与GT框IOU大于pos_iou_thresh的作为正样本
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # 获取规定正样本数与实际正样本数中较小的
        pos_roi_per_this_image = int(min(self.pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size = pos_roi_per_this_image, replace = False)
        
        # 将anchor框与GT框大于neg_iou_thresh_low小于neg_iou_thresh_high的作为负样本
        # 正负样本总数保持在n_sample
        neg_index = np.where((max_iou < self.neg_iou_thresh_high) & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size = neg_roi_per_this_image, replace = False)

        keep_index = np.append(pos_index, neg_index)  # 合并所有的正负样本

        sample_roi = roi[keep_index]  # 取出roi
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]
        
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # 负样本全部设置为背景类
        return sample_roi, gt_roi_loc, gt_roi_label

class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn  # 网络主体架构
        self.optimizer = optimizer  # 优化器

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]
    
    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        # pred_loc 预测参数
        # gt_loc 真实参数
        # gt_label 真实标签， 标式正样本或负样本或忽略· 
        # 1 正样本， 0 负样本， -1 忽略 位置回归参数仅对正样本计算loss

        pred_loc = pred_loc[gt_label > 0]  # 仅考虑正样本  
        gt_loc = gt_loc[gt_label > 0]  # 仅考虑正样本

        sigma_squared = sigma ** 2
        regression_diff = (gt_loc - pred_loc)  # 真实标签与预测标签的差距
        regression_diff = regression_diff.abs()  # 取绝对值
        regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared)  # MSEloss
        regression_loss = regression_loss.sum()
        num_pos = (gt_label > 0).sum().float()  # 正样本数量

        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss
    
    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]  # 每个batch内的样本数量
        img_size = imgs.shape[2:]  # 每张图片的高宽

        base_feature = self.faster_rcnn.extractor(imgs)  # 经过基础特征提取网络

        # 利用rpn网络获得坐标、尺寸调整参数，得分，调整及筛选后的anchor框，以及所有anchor框
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(base_feature, img_size, scale)

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        
        # 对batch内每个样本进行处理
        for i in range(n):
            bbox = bboxes[i]  # (num_gt, 4) (xmin, ymin, xmax, ymax) num_gt为每张图包含的真实框数量
            label = labels[i]  # (num_gt, )
            rpn_loc = rpn_locs[i]  #（h * w * 9， 4）  h, w 为下采样后的特征图的高和宽
            rpn_score = rpn_scores[i]  # (h * w * 9, 2)
            roi = rois[roi_indices == i]  # (n_post_nms, 4)  n_post_nms 为经过nms后的roi区域
            feature = base_feature[i] 
            
            # 选取rpn网络正负样本及正样本对应的真实参数
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor)
            gt_rpn_loc = torch.Tensor(gt_rpn_loc)  # 转化为tensor
            gt_rpn_label = torch.Tensor(gt_rpn_label).long()  # 转化为tensor

            # 转化为cuda格式
            if rpn_loc.is_cuda:
                gt_rpn_loc = gt_rpn_loc.cuda()
                gt_rpn_label = gt_rpn_label.cuda()
            
            # 分别计算建议框网络的回归损失和分类损失
            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index = -1)  # 忽略-1

            # 利用真实框和建议框获得classifier应该的预测结果
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            # 转化为tensor
            sample_roi = torch.Tensor(sample_roi)  # 筛选后的roi
            gt_roi_loc = torch.Tensor(gt_roi_loc)  # 应该与GT框的差距
            gt_roi_label = torch.Tensor(gt_roi_label).long()  # 应该的类别
            sample_roi_index = torch.zeros(len(sample_roi))  # 初始化roi的索引,因为按单张图片计算，所以全部为0

            if feature.is_cuda:
                sample_roi = sample_roi.cuda()
                sample_roi_index = sample_roi_index.cuda()
                gt_roi_loc = gt_roi_loc.cuda()
                gt_roi_label = gt_roi_label.cuda()
            
            roi_cls_loc, roi_score = self.faster_rcnn.head(torch.unsqueeze(feature, 0), sample_roi, sample_roi_index, img_size)

            # 根据建议框的种类，去除对应的回归结果
            n_sample = roi_cls_loc.size()[1]  # 保留的roi个数
            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4) # (n_sample, num_classes + 1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label]  # 取出GT类别对应的loc参数

            # 计算classifier网络的回归损失与分类损失
            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score[0], gt_roi_label)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss
            roi_loc_loss_all += roi_loc_loss
            roi_cls_loss_all += roi_cls_loss

        losses = [rpn_loc_loss_all / n, rpn_cls_loss_all / n, roi_loc_loss_all / n, roi_cls_loss_all / n]
        losses = losses + [sum(losses)]
        return losses
    
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses[-1].backward()
        self.optimizer.step()
        return losses