import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input


###### 定义数据读取及加载 ######


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape = [600, 600], train = True):
        super(FRCNNDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape  # [高, 宽]
        self.length = len(annotation_lines)
        self.train = train
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):

        image, box = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype = np.float32)), (2, 0, 1))  # 调换维度顺序
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            box_data[:len(box)] = box
        
        box = box_data[:, :4]  # 每个GT框的左上角及右下角坐标
        label = box_data[:, -1]  # 每个GT框对应的类别
        return image, box, label

    # 生成a-b范围内的随机数
    def rand(self, a = 0, b = 1):
        return np.random.rand() * (b - a) + a
    
    def get_random_data(self, annotation_line, input_shape, jitter = .3, hue = .1, sat = 1.5, val = 1.5, random = True):
        line = annotation_line.split()         
        image = Image.open(line[0])  # 读取图像
        image = cvtColor(image)  # 转化为RGB图像形式

        iw, ih = image.size  # 原始图像的宽和高
        h, w = input_shape  # 要求输入的宽和高

        # 读取box相关参数
        # xmin, ymin, xmax, ymax, cls_id
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:  
            # 处于测试状态
            scale = min(w / iw, h / ih)
            nw = int(scale * iw)
            nh = int(scale * ih)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            
            # 将图像长边缩放至目标尺寸， 短边缺少的部分用灰色填充
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                # 调整框的大小
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            
                # 将超出边界的值都整合至边界内
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h

                # 计算框的宽度及长度
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]

                # 剔除宽和高小于1像素的框
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box
        
        # 对图像进行缩放并进行长和宽的扭曲
        
        # 生成新的宽高比
        # 生成新的宽高比
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)  # 生成随机尺度
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像缺少的部分补上灰边
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转

        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            if flip: 
                box[:, [0, 2]] = w - box[:, [2, 0]]
            
            # 将超出边界的值都整合至边界内
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h

            # 计算框的宽度及长度
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]

            # 剔除宽和高小于1像素的框
            box = box[np.logical_and(box_w > 1, box_h > 1)]
    
        return image_data, box

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = np.array(images)
    return images, bboxes, labels