import torch
from opts_and_utils import _transpose_and_gather_feat,AverageMeter

class RegL1Loss(torch.nn.Module):
  def __init__(self):
    super().__init__()

  #首先，要知道output和target分别是out和gt
  #那么，mask和ind肯定是辅助定位到只有中心点处才计算wh和reg的loss，别的不管
  def forward(self, output, mask, ind, target):
    #print(ind.shape)#torch.Size([1, 50]) 咦，这个128是我自己设置的，128好像挺多的，改成50吧
    #ind[k] = ct_int[1] * output_w + ct_int[0]  #这个代码说明，ind是把中心点坐标储藏成了整数的形式

    #然后得到50个坐标对应的hw或reg值
    #例如，如果gt只有一个目标，那个ind就会为【xxxx，0,0,0,0.......】
    #然后第一个xxxx会读到一个坐标的hw或reg值，剩下的0都是同一个坐标
    #所以得到的pred是【xxxx，uuu，uuu，uuu，uuu，...】后面的都是相同值
    #pred被还原成了坐标形式 2*50矩阵
    pred = _transpose_and_gather_feat(output, ind)

    #进而，通过mask告诉我，到底要前几个数
    mask = mask.unsqueeze(2).expand_as(pred).float()

    #然后就能算l1 loss了！！！！！！
    loss = torch.nn.functional.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''

  #pos就是所有的中心点
  pos_inds = gt.eq(1).float()
  #print(torch.sum(pos_inds))

  #所有非中心点
  neg_inds = gt.lt(1).float()

  #贝塔乘以1-gt
  neg_weights = torch.pow(1 - gt, 4)
  loss = 0

  #公式的两部分
  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  #loss要除以gt里的目标数，如果gt目标数为0，那就除以1
  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class FocalLoss(torch.nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super().__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class CtdetLoss(torch.nn.Module):
  def __init__(self):
    super().__init__()

  #可以看到，这里输入了个batch参数
  def forward(self, outputs, batch):
    hm_loss, wh_loss, off_loss = 0, 0, 0

    output = outputs[0]
    output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1-1e-4)

    #构建hm_loss层、wh_loss层、off_loss层
    hm_loss += FocalLoss()(output['hm'], batch['hm'])


    wh_loss += RegL1Loss()(output['wh'], batch['reg_mask'],batch['ind'], batch['wh'])
    off_loss += RegL1Loss()(output['reg'], batch['reg_mask'],batch['ind'], batch['reg'])
        
    loss = hm_loss + 0.1 * wh_loss + off_loss

    #用来显示各种loss是多少
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats

class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats


class CtdetTrainer():
  def __init__(self, model, optimizer):
    self.optimizer = optimizer

    #self.loss是个model，输入【out和gt】，输出loss
    self.loss_stats, self.loss = self._get_losses()

    #把两个model串起来，最终得到的model_with_loss，输入是数据和标签，输出是loss
    self.model_with_loss = ModelWithLoss(model, self.loss)

    self.all_val_loss_list=[]
  
  def _get_losses(self):

    #loss是个model，输入【out和gt】，输出loss
    loss = CtdetLoss()
    return ['loss', 'hm_loss', 'wh_loss', 'off_loss'], loss

  def run_epoch(self, phase, epoch, data_loader):

    #每过完一个val_epoch清零一次
    self.each_val_epoch_mean_loss = 0
    model_with_loss = self.model_with_loss


    #所以最终得到的用来训练的网络，和那个yolov3那个理论基本是一致的
    #输入是，含数据和标签的batch变量
    # 输出是[{长度三的那个输出的字典}]、总loss、分loss

    # 在训练模型时会在前面加上：
    # model.train()
    # 在测试模型时在前面使用：
    # model.eval()Normalization和Dropout。
    # model.train() ：启用BatchNormalization和Dropout
    # model.eval() ：不启用BatchNormalization和Dropout
    if phase == 'train':
      print('\n现在是epoch:',epoch,'\n')
      model_with_loss.train()
    else:
      model_with_loss.eval()

    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader)


    ################下方是训练与打印信息#########

    #batch就是data_loader的一部分
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break

      _,loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()

      if phase == 'train':

        #模板
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      show_str = '{phase}: [epoch:{0}][{1}/{2}] '.format(epoch, iter_id, num_iters, phase=phase)

      for l in avg_loss_stats:
        avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
        show_str = show_str + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if phase == 'train':
        print(show_str)
      if phase == 'val':
        self.each_val_epoch_mean_loss +=avg_loss_stats['loss'].avg
        del loss, loss_stats
    if phase == 'val':
      print('现在的平均val_loss是：',self.each_val_epoch_mean_loss/num_iters)
      self.all_val_loss_list.append(self.each_val_epoch_mean_loss/num_iters)









