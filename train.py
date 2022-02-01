import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch
from nets.frcnn_training import FasterRCNNTrainer
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### 训练Faster R-CNN网络 ######


class train_frcnn():
    def __init__(self):
        super(train_frcnn, self).__init__()

        classes_path = 'model_data/voc_classes.txt'
        model_path = 'model_data/frcnn_weights.pth'

        train_annotation_path = 'VOCdevkit/VOC2007/ImageSets/Main/2007_train.txt'  # 训练集标签文件存储路径
        test_annotation_path = 'VOCdevkit/VOC2007/ImageSets/Main/2007_test.txt'  # 测试集标签文件存储路径

        anchors_size = [8, 16, 32]

        self.input_shape = [600, 600]  # 输入尺寸 
        self.class_names, self.num_classes = get_classes(classes_path)  # 获取类别名称及类别哦那个数

        # 创建Faster R-CNN模型
        model = FasterRCNN(self.num_classes, anchor_scales = anchors_size)
        print('Load Weights from {}.'.format(model_path))

        model_dict = model.state_dict()  # 模型参数
        pretrained_dict = torch.load(model_path, map_location = device)
        # 替换key相同且shape相同的值
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)  # 更新参数
        model.load_state_dict(model_dict)  # 加载参数

        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.to(device)
        
        self.model = model

        with open(train_annotation_path, 'r', encoding = 'utf-8') as f:
            self.train_lines = f.readlines()  # 读取训练集数据
        with open(test_annotation_path, 'r', encoding = 'utf-8') as f:
            self.test_lines = f.readlines()  # 读取测试集数据
        
        self.loss_test_min = 1e9  # 初始话最小测试集loss
    
    def train(self, batch_size, learning_rate, start_epoch, end_epoch, Freeze = False):

        # 定义优化器
        optimizer = optim.Adam(self.model.parameters(), learning_rate, weight_decay = 5e-4)
        
        # 学习率下降策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)

        # 定义训练集与测试集
        train_dataset = FRCNNDataset(self.train_lines, self.input_shape, train = True)
        test_dataset = FRCNNDataset(self.test_lines, self.input_shape, train = False)
        train_data = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 4,
                                  pin_memory = True, drop_last = True, collate_fn = frcnn_dataset_collate)
        test_data = DataLoader(test_dataset, shuffle = True, batch_size = batch_size,num_workers = 4,
                                  pin_memory = True, drop_last = True, collate_fn = frcnn_dataset_collate)
        
        # 冻结backbone参数
        if Freeze:
            for param in self.model.extractor.parameters():
                param.requires_grad = False
        else:
            for param in self.model.extractor.parameters():
                param.requires_grad = True
        
        train_util = FasterRCNNTrainer(self.model, optimizer)
        
        # 开始训练
        for epoch in range(start_epoch, end_epoch):
            print('Epoch: ', epoch)
            train_loss, test_loss = fit_one_epoch(self.model, train_util, optimizer, train_data, test_data, device)
            lr_scheduler.step()

            # 若测试集loss小于当前极小值，保存当前模型
            if test_loss < self.loss_test_min:
                self.loss_test_min = test_loss
                torch.save(self.model.state_dict(), 'faster_rcnn.pth')
    
    def total_train(self):

        # 首先进行backbone冻结训练
        Freeze_batch_size = 4
        Freee_lr = 1e-4
        Init_epoch = 0
        Freeze_epoch = 50

        self.train(Freeze_batch_size,Freee_lr, Init_epoch, Freeze_epoch, Freeze = True)

        # 解冻backbone进行训练
        batch_size = 2
        learning_rate = 1e-5
        end_epoch = 100

        self.train(batch_size, learning_rate, Freeze_epoch, end_epoch, Freeze = False)

if __name__ == "__main__":
    train = train_frcnn()
    train.total_train()