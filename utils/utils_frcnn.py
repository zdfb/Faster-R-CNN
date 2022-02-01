import os
import torch
import numpy as np
import torch.nn as nn
from nets.frcnn import FasterRCNN
from utils.utils_bbox import DecodeBox
from PIL import Image, ImageDraw, ImageFont
from utils.utils import cvtColor, get_classes, get_new_img_size, resize_image, preprocess_input, ncolors


###### 解析模型，生成最终结果 ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FRCNN(object):
    def __init__(self):
        super(FRCNN, self).__init__()

        model_path = 'model_data/frcnn_weights.pth'  # 模型存储路径
        classes_path = 'model_data/voc_classes.txt'  # 类别信息存储路径

        self.confidence = 0.5  # 置信率初筛阈值
        self.nms_iou = 0.3  # 非极大值抑制IOU阈值
        self.anchors_size = [8, 16, 32]

        # 获取种类名及数量
        self.class_names, self.num_classes = get_classes(classes_path)

        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        self.std = self.std.to(device)

        self.colors = ncolors(self.num_classes)

        self.bbox_util = DecodeBox(self.std, self.num_classes)

        model = FasterRCNN(self.num_classes, mode = 'predicr', anchor_scales= self.anchors_size)

        model.load_state_dict(torch.load(model_path, map_location = device))
        model = model.eval()

        self.model = model.to(device)
    
    # 检测图片
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])  # 输入图像的宽和高
        input_shape = get_new_img_size(image_shape[0], image_shape[1])  # 将短边resize为600

        image = cvtColor(image)  # 将输入图像转化为RGB形式

        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        # 对输入图像进行预处理
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype = 'float32')),(2, 0, 1)), 0)

        with torch.no_grad():
            image_ = torch.from_numpy(image_data)
            image_ = image_.to(device)
            
            # roi_cls_locs 建议框的调整参数
            # roi_scores 建议框的种类得分
            # rois 未微调的建议框坐标
            roi_cls_locs, roi_scores, rois, _ = self.model(image_)

            # 对上述结果进行解码，获得最终输出建议框
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = self.nms_iou, confidence = self.confidence)

            if results[0] is None:
                return image
            
            top_label = np.array(results[0][:, 5], dtype = 'int32')  # 预测类别
            top_conf = results[0][:, 4]  # 预测置信率
            top_boxes = results[0][:, :4]  # 预测框位置 (num_bbox, (ymin, xmin, ymax, xmax))
        
        # 绘制图像上的标注框
        font_size = np.floor(2e-2 * image.size[1]).astype('int32')  # 定义字体大小
        font = ImageFont.truetype(font = 'model_data/simhei.ttf', size = font_size)  # 定义字体样式

        for index, class_id in list(enumerate(top_label)):
            predicted_class = self.class_names[int(class_id)]  # 取出预测类别名称

            box = top_boxes[index]  # 预测框的位置信息 （ymin, xmin, ymax, xmax）
            score = top_conf[index]  # 预测框的置信度

            ymin, xmin, ymax, xmax = box  # 取出坐标详细信息

            # 标签内容
            label_text = '{}{:.2f}'.format(predicted_class, score)

            # 绘制图像
            draw = ImageDraw.Draw(image)

            # 获取标签区域大小
            label_size = draw.textsize(label_text, font)

            # 绘制标签包围框
            draw.rectangle((xmin, ymin - label_size[1], xmin + label_size[0], ymin), fill = self.colors[class_id])
            # 绘制目标框
            draw.rectangle((xmin, ymin, xmax, ymax), outline = self.colors[class_id], width = 3)
            # 绘制标签
            draw.text((xmin, ymin - label_size[1]), label_text, fill = (255, 255, 255), font=font)
            del draw
        return image