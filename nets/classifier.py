import torch
from torch import nn
from torchvision.ops import RoIPool


###### 定义分类头 ######


# 定义ROIheads
class Resnet50ROIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50ROIHead, self).__init__()
        self.classifier = classifier  # 后续的特征提取

        self.cls_loc = nn.Linear(2048, n_class * 4)  # 对ROIPooling后的结果进行回归预测
        self.score = nn.Linear(2048, n_class)  # 分类器
        
        # ROI头
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape  # (n, 1024, 38, 38)

        # 放置在相同的环境中
        roi_indices = roi_indices.to(x.device)  # (n * n_post_nms, )
        rois = rois.to(x.device)  # (n * n_post_nms, 4)

        rois_feature_map = torch.zeros_like(rois) # (n * n_post_nms, 4)

        # 将roi框对应至featuremap上
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]
        
        # 将表示样本在batch中顺序的与Roi在特征图上的映射合并在一起 # (n * n_post_nms, 5)
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim = 1)

        # 利用建议框对特征层进行截取 (n * n_post_nms, 1024, 14, 14)
        pool = self.roi(x, indices_and_rois)

        # 利用classifier网络进行特征提取
        # classifier层, 经过layer4，转化为 (n * n_post_nms. 2048, 7, 7)
        # 经过avgpool, 转化为(n * n_post_nms, 2048, 1, 1)
        fc7 = self.classifier(pool) 
        fc7 = fc7.view(fc7.size(0), -1)  # (n * n_post_nms, 2048)

        roi_cls_locs = self.cls_loc(fc7)  # (n * n_post_nms, 4 * num_class)
        roi_scores = self.score(fc7)  # (n * n_post_nms, num_class)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))  # (n, n_post_nms, 4 * num_class)
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))  # (n, n_post_nms, num_class)
        return roi_cls_locs, roi_scores