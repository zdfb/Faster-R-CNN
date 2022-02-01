import math
import torch
import torch.nn as nn


###### 定义Resnet50基础特征提取网络 ######


# 定义Bottleneck
class Bottleneck(nn.Module):
    expansion = 4  # 通道放大倍数
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x  # 原始输入特征

        out = self.conv1(x)  # 修改通道数，若stride == 1，特征图尺寸不变，若stride == 2， 特征图尺寸减半
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # 提取上下文特征，特征图尺寸不变
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 扩展通道数
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # 若修改了特征图尺寸，原输入特征图应下采样
        
        out += residual  # skip-connect
        out = self.relu(out)  # 先相加再relu

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        # 600, 600, 3 -> 300, 300, 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)

        # 300, 300, 64 -> 150, 150, 64
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)

        # 150, 150, 64 -> 150, 150, 256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150, 150, 256 -> 75, 75, 512
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        # 75, 75, 512 -> 38, 38, 1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        # 38, 38, 1024 -> 19, 19, 2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.avgpool = nn.AvgPool2d(7)

    
    # 构建中间提取层
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None  # 初始化不设置下采样
        
        # 若存在下采样倍数或者输出通道数不匹配
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

def resnet50():
    model = ResNet(Bottleneck, [3,4,6,3])
    # 特征提取部分
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # 分类器部分
    classifier  = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier