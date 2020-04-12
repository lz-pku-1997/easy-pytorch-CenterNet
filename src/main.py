# centernet 简化版，适用 windows + cpu
# object as point那篇
# pytorch新手，注释比较多，新冠期间在家无设备，使用CPU，无多线程
# pytorch版本1.0.0
# pycocotools windows按这个链接就好    http://www.mamicode.com/info-detail-2660241.html


import os
import torch
from msra_resnet import get_pose_net
from coco_ctdet import COCO
from CtdetTrainer import CtdetTrainer
from opts_and_utils import all_opts
opts=all_opts()

def save_model(path, epoch, model, optimizer=None):

  state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)


#给每个val epoch求个平均值存入list
show_mean_val_loss_list=[]


def main():
  # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括
  # batch_size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
  torch.backends.cudnn.benchmark = True

  print('\n正在创建resnet+上采样+三个输出层的模型...')
  #得到了带上采样且带三个head的model
  model = get_pose_net(opts.res_layers, opts.classes)

  # 固定句式
  optimizer = torch.optim.Adam(model.parameters(), opts.lr)

  #得到了CtdetTrainer类，里面含了 带上loss的model
  trainer = CtdetTrainer(model, optimizer)

  #这个keras应该一样，处理都藏在了DataLoader里
  # 默认参数
  # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
  #            batch_sampler=None, num_workers=0, collate_fn=None,
  #            pin_memory=False, drop_last=False, timeout=0,
  #            worker_init_fn=None)
  # drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
  #pin_memory：是否将数据保存在pin_memory区，pin_memory中的数据转到GPU会快一些
  #num_workers：使用多进程加载的进程数，0代表不使用多进程
  print('\n开始加载训练和测试的数据咯\n')
  val_loader = torch.utils.data.DataLoader(COCO('val'), batch_size=opts.batch_size,num_workers=1)

  #Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
  #这个COCO类就是我继承的Dataset类
  train_loader = torch.utils.data.DataLoader(COCO('train'), batch_size=opts.batch_size,shuffle=True)

  print('开始训练咯...')
  for epoch in range(1, opts.num_epochs + 1):
    trainer.run_epoch('train',epoch, train_loader)
    save_model(os.path.join(os.path.dirname(__file__), 'weight', 'model_{}.pth'.format(epoch)),epoch, model, optimizer)
    print('保存成功，去看看')
    with torch.no_grad():
      trainer.run_epoch('val',epoch, val_loader)

  #每个epochs的val_loss列表
  print('所有的val_loss列表：\n',trainer.all_val_loss_list)

if __name__ == '__main__':
  main()