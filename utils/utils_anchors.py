import numpy as np


###### 生成anchors ######


# 生成每个点都anchor
def generate_anchor_base(base_size = 16, ratios=[0.5, 1, 2], anchor_scales = [8, 16, 32]):
    # 生成9个anchor (9,4)
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype = np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])  # 获取该尺度下的高
            w = base_size * anchor_scales[j] * np.sqrt(1./ratios[i])  # 获取该尺度下的宽

            index = i * len(anchor_scales) + j  # 获取anchorbase的索引
            anchor_base[index, 0] = -h / 2.
            anchor_base[index, 1] = -w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2. 
    return anchor_base

# 将每个anchor对应在特征图上
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 组成网格
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)

    A = anchor_base.shape[0]  # 9个anchor
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))  # 利用numpy的传递性 转化为（K, A, 4）
    anchor = anchor.reshape((K * A, 4)).astype(np.float)  # 尺寸为特征图（h * w *9, 4）
    return anchor