import time
import torch
import numpy as np


###### 训练一个epoch ######


def fit_one_epoch(model, train_util, optimizer, train_data, test_data, device):

    start_time = time.time()  # 获取当前时间
    model.train()  # 训练过程

    total_loss = []  # 总体loss

    for step, data in enumerate(train_data):
        images, boxes, labels = data[0], data[1], data[2]  # 取出图片、GT框信息及对应的标签

        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
        
        rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1)  # 开始训练

        total_loss.append(total.item())

        # 画进度条
        rate = (step + 1) / len(train_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, total), end="")
    print()

    model.eval()  # 测试过程

    test_total_loss = []  # 总体loss
    test_rpn_loc_loss = []  # rpn定位loss
    test_rpn_cls_loss = []  # rpn分类loss
    test_roi_loc_loss = []  # roi定位loss
    test_roi_cls_loss = []  # roi分类loss

    for step, data in enumerate(test_data):
        images, boxes, labels = data[0], data[1],data[2]  # 取出图片、GT框信息及对应的标签
        
        with torch.no_grad():
            images = torch.from_numpy(images).type(torch.FloatTensor).to(device)

            train_util.optimizer.zero_grad()
            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.forward(images, boxes, labels, 1)

            test_total_loss.append(total.item())
            test_rpn_loc_loss.append(rpn_loc.item())
            test_rpn_cls_loss.append(rpn_cls.item())
            test_roi_loc_loss.append(roi_loc.item())
            test_roi_cls_loss.append(roi_cls.item())
        
        # 画进度条
        rate = (step + 1) / len(test_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtest loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, total), end="")
    print()

    train_loss = np.mean(total_loss)  # 该epoch总的训练loss
    test_loss = np.mean(test_total_loss)  # 该epoch总的测试loss

    rpn_loc_loss = np.mean(test_rpn_loc_loss)
    rpn_cls_loss = np.mean(test_rpn_cls_loss)
    roi_loc_loss = np.mean(test_roi_loc_loss)
    roi_cls_loss = np.mean(test_roi_cls_loss)

    stop_time = time.time()  # 获取当前时间
    
    print('total_train_loss: %.3f, total_test_loss: %.3f, epoch_time: %.3f.'%(train_loss, test_loss, stop_time - start_time))
    print('rpn_loc_loss : %.3f, rpn_cls_loss: %.3f, roi_loc_loss: %.3f, roi_cls_loss: %.3f.'%(rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss))
    return train_loss, test_loss