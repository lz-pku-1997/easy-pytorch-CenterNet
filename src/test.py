#测试推理部分核心代码
import cv2
import os
import numpy as np
import torch
from msra_resnet import get_pose_net
from opts_and_utils import _gather_feat, _transpose_and_gather_feat,all_opts
opts=all_opts()



def _nms(heat):
  hmax = torch.nn.functional.max_pool2d(heat, (3, 3), stride=1, padding=1)
  keep = (hmax == heat).float()
  return heat * keep


def _topk(scores, K):
  # hm 四维张量
  batch, cat, height, width = scores.size()

  # topk即kop_k的意思
  # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
  # 沿给定dim维度返回输入张量input中 k 个最大值
  # view中参数中的-1就代表这个位置由其他位置的数字来推断
  # 所以最终是排出  每张图中的每个类的前50名
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
  # print(topk_inds.shape)#torch.Size([1, 1, 50])

  # 这个id也隐藏了 类别层的信息，需要取余去掉类别层信息，只含xy位置信息
  topk_inds = topk_inds % (height * width)

  # 得到x、y信息
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  # 在所有class间再整合一次，最终得到整张图的前50
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  # print(topk_score.shape)#torch.Size([1, 50]) 即（batch，50）

  # 存着每张图 top 50的类别号#torch.Size([1, 50])  即（batch，50）
  topk_clses = (topk_ind / K).int()

  # 取得最终所需的前50个信息
  topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg, K):
  batch, cat, height, width = heat.size()

  # 这个就是论文里说的，hm里只留下比周围8个点都大的那些点
  heat = _nms(heat)

  # 得到每张图scores最高的前50个点的 scores、坐标信息、类别、坐标信息中的y坐标、坐标信息中的x坐标
  scores, inds, clses, ys, xs = _topk(heat, K=K)

  # 取得每张图前50名scores对应位置的reg值
  reg = _transpose_and_gather_feat(reg, inds)
  # print(reg.shape) #torch.Size([1, 50, 2])

  reg = reg.view(batch, K, 2)

  # print(reg) #都是正数
  # 加上reg值
  xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
  ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

  wh = _transpose_and_gather_feat(wh, inds)
  wh = wh.view(batch, K, 2)
  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2,
                      ys + wh[..., 1:2] / 2], dim=2)
  detections = torch.cat([bboxes, scores, clses], dim=2)
  return detections


def load_model(model, model_path):
  checkpoint = torch.load(model_path)
  print('\n加载权重 {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model.load_state_dict(state_dict, strict=False)
  return model


class CtdetDetector():
  def __init__(self):
    print('\n正在创建模型...')
    self.model = get_pose_net(opts.res_layers, opts.classes,train_or_test='test')

    #官方代码中加载模型的函数，应该是保存的模型引入了额外的一些参数
    self.model = load_model(self.model, opts.load_model)
    #一种常见用法,在这会出错
    # pretrained_dict = torch.load(opt_load_model)
    # model_dict = self.model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # self.model.load_state_dict(model_dict)

    # model.eval() ：不启用BatchNormalization和Dropout
    self.model.eval()

    self.pause = True
  
  def process(self, images):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg']

      #通过最后三层得到[bbox、置信度、类别]
      dets = ctdet_decode(hm, wh, reg, opts.max_object)

    return dets

  def run(self, image_path):

    #初步预处理
    image = cv2.imread(image_path)
    temp_image = ((image / 255. - opts.mean) / opts.std).astype(np.float32)
    temp_image = temp_image.transpose(2, 0, 1)[np.newaxis, :]
    images= torch.from_numpy(temp_image)

    dets = self.process(images)

    #随便写个显示，看看结果
    for det in np.array(dets[-1]):
      if det[4]>0.3:
        det[0],det[1],det[2],det[3]=4*det[0],4*det[1],4*det[2],4*det[3]
        cv2.rectangle(image, (det[0], det[1]), (det[2], det[3]),(0,0,255),2)
    cv2.imshow('temp',image)
    cv2.waitKey()


#防止文件夹里还有别的文件，以至于报错
image_ext = ['jpg', 'jpeg', 'png', 'webp']
def demo():
  detector = CtdetDetector()

  if os.path.isdir(opts.demo): #比如这里输入了c.png,然后判断不是个路径
    image_names = []
    ls = os.listdir(opts.demo)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(opts.demo, file_name))
  else:
    image_names = [opts.demo]

  for (image_name) in image_names:

    #推理
    detector.run(image_name)

if __name__ == '__main__':
  demo()


