import numpy as np
import os

class all_opts():
    def __init__(self):

        #训练和测试共用的配置
        self.res_layers=152                         #resnet层数，可选50,101,152
        self.classes=1                              #类别数
        self.class_name = ['cancer_cell', ]         #类别名们
        self.mean = np.array([0.8292903,0.74784886,0.80975633],dtype=np.float32).reshape(1, 1, 3)#数据集的归一化均值
        self.std = np.array([0.1553852,0.20757463,0.16293081],dtype=np.float32).reshape(1, 1, 3) #数据集的归一化方差
        self.max_object=50                          #一张图里最多有多少个目标，可以设大，不能设小
        self.img_size=256
        self.data_dir = os.getcwd()+'\\..\\data\\coco'#数据集路径

        #训练部分专用的配置
        self.lr = 0.00008                           #lr
        self.batch_size = 1                         #batch_size
        self.num_epochs = 15                        #epochs

        #测试部分专用的配置
        self.load_model='./weight/model_15.pth'     #测试用的权重文件的路径
        self.demo = '..\\data\\coco\\val2017'       #测试存放图片的位置,可以输入一个文件夹、也可以输入一张图片,这里默认的地址为验证集




#以下均是工具人函数
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1  # 例如半径是2时，直径是5
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    # print ('\n',masked_gaussian)

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug

        # 如果高斯图重叠，重叠的点取最大值就行
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count