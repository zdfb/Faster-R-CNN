import torch
import numpy as np
from torchvision.ops import nms
from torch.nn import functional as F


###### 用于生成目标anchor框 ######


# 将anchor尺寸根据输出参数进行调整
def loc2bbox(src_bbox, loc):
    # src_bbox (h * w * 9, 4)
    # loc (h * w * 9, 4)
    # 判断输入是否为0
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype = loc.dtype)
    
    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)  # 每个anchor框的宽 （h * w * 9, 1）
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)  # 每个anchor的高 
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width  # 中心点x坐标
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height # 中心点y坐标

    dx = loc[:, 0::4]  # x调整参数
    dy = loc[:, 1::4]  # y调整参数
    dw = loc[:, 2::4]  # 宽度调整参数
    dh = loc[:, 3::4]  # 高度调整参数

    ctr_x = dx * src_width + src_ctr_x  # 调整后的中心x参数
    ctr_y = dy * src_height + src_ctr_y  # 调整后的中心y参数
    w = torch.exp(dw) * src_width  # 调整后的框宽度
    h = torch.exp(dh) * src_height  # 调整后的框高度

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w  # 左上角x坐标
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h  # 左上角y坐标
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w  # 右下角x坐标
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h  # 右下脚y坐标

    return dst_bbox

# 解码输出结果
class DecodeBox():
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes + 1
    
    def frcnn_correct_boxes(self, box_xy, box_wh, image_shape):
        # 调换x轴与y轴
        box_yx = box_xy[..., ::-1]  
        box_hw = box_wh[..., ::-1]
        image_shape = np.array(image_shape)
        
        # 转化为ymin, xymin, ymax, xmax的形式
        box_mins = box_yx - (box_hw / 2.) 
        box_maxs = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[..., 0:1], box_maxs[..., 1:2]], axis = -1)

        # 应该为 boxes * input_shape * image_shape / input_shape == boxes * image_shape
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)

        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []  # 最终结果
        bs = len(roi_cls_locs)  # 每个batch内包含的样本个数

        rois = rois.view((bs, -1, 4))  # (bs, num_rois, 4)

        # 对每一张图片进行处理
        for i in range(bs):
            
            roi_cls_loc = roi_cls_locs[i] * self.std  #  调整回归参数
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])  # (num_rois, num_classes, 4)
            
            # (num_rois, 4) -> (num_rois, 1, 4) -> (num_rois, num_classes, 4)
            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)

            # 将每一个roi都根据对应类别参数调整 （num_rois * num_classes, 4)
            cls_bbox = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            # (num_rois * num_classes, 4) -> (num_rois, num_classes, 4)
            cls_bbox = cls_bbox.view([-1, (self.num_classes), 4]) # (num_rois, num_classes, 4)

            # 对预测框进行归一化，调整到0～1之间
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]  # 除以输入宽度，进行归一化
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]  # 除以输入高度，进行归一化

            roi_score = roi_scores[i]  # (num_rois, num_classes)
            prob = F.softmax(roi_score, dim = -1)  # 使用softmax输出值转化为0～1的概率值

            results.append([])
            # 对每一个类别进行处理
            # 第一类是背景类
            for c in range(1, self.num_classes):
                c_confs = prob[:, c]  # 取出属于该类的框的置信度
                c_confs_m = c_confs > confidence  # 取出大于阈值的部分

                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = cls_bbox[c_confs_m, c]  # 取出对应的第c类，置信率大于阈值的框
                    confs_to_process = c_confs[c_confs_m]  # 取出对应的阈值

                    # 执行非极大值抑制
                    keep = nms(boxes_to_process, confs_to_process, nms_iou)

                    # 取出非极大值抑制后的结果
                    good_boxes = boxes_to_process[keep]  # 处理后的框
                    confs = confs_to_process[keep][:, None]  # 处理后的置信率
                    # 每个框对应的标签
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))

                    # 将label, 置信度，框的坐标信息进行堆叠
                    c_pred = torch.cat((good_boxes, confs, labels), dim = 1).cpu().numpy()  # (n_boxes, 6)

                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])  # 将所有结果转化为numpy格式
                # 转化为中点及长宽形式
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, image_shape)
        return results