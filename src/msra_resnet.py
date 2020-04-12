import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo



BN_MOMENTUM = 0.1
#resnet的一些参数
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
resnet_spec = {50: [3, 4, 6, 3],101: [3, 4, 23, 3],152: [3, 8, 36, 3]}


#inplanes---planes---4*planes
class Bottleneck(nn.Module):
    expansion = 4

    # Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #需要下采样的层是在Bottleneck里加上下采样的卷积，而非改变某个卷积的stride
        if self.downsample is not None:
            residual = self.downsample(x)

        #每个Bottleneck里都有一个残差块
        out += residual
        out = self.relu(out)

        return out



'''
官方代码head的定义
elif opt.task == 'ctdet':
    opt.heads = {'hm': opt.num_classes,'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
官方默认cat_spec_wh是False，所以'wh': 2

if opt.reg_offset:
    opt.heads.update({'reg': 2})
官方--not_reg_offset参数为False，opt.reg_offset = not opt.not_reg_offset，所以reg_offset是True

所以综上所述，官方的默认heads参数为{'hm': opt.num_classes,'wh': 2,'reg': 2}



官方head_conv default是-1，所以根据下方的官方代码会变成64  
if opt.head_conv == -1:
    opt.head_conv = 256 if 'dla' in opt.arch else 64
'''

'''默认
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
'''
class PoseResNet(nn.Module):
    def __init__(self,layers, heads_classes, head_conv, **kwargs):

        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = {'hm': heads_classes, 'wh': 2, 'reg': 2}

        super().__init__()

        #resnet的第一部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #con2~con5
        # res152传layers[i]为[3, 8, 36, 3]
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # used for deconv layers
        #就是做三次上采样
        self.deconv_layers = self._make_deconv_layer()

        #最后三层
        self.hm = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),nn.ReLU(inplace=True),nn.Conv2d(head_conv, heads_classes, kernel_size=1))
        self.wh = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),nn.ReLU(inplace=True),nn.Conv2d(head_conv, 2, kernel_size=1))
        self.reg = nn.Sequential(nn.Conv2d(256, head_conv, kernel_size=3, padding=1),nn.ReLU(inplace=True),nn.Conv2d(head_conv, 2, kernel_size=1))

    #res152传blocks为[3, 8, 36, 3]
    def _make_layer(self, planes, blocks, stride=1):

        #每组的第一个要考虑是否下采样
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * Bottleneck.expansion, momentum=BN_MOMENTUM),
        )

        layers = []

        # Bottleneck inplanes---planes---4*planes
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))

        #嗯嗯~四倍咯
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self):
        layers = []
        for i in range(3):
            planes = 256

            #if i==0:
                #print('第0次的inplanes是：',self.inplanes)#第0次的inplanes是resnet152输出的2048
            #默认torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            layers.append(nn.ConvTranspose2d(self.inplanes,planes,4,stride=2,padding=1,bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes


        #' * ' 的作用
        #第一种：用在动态参数前，打包多个参数并将其转化为元组，例如：
        # def func(*args):
        #     print(args)
        #
        # func(1, 2, 3)  # (1, 2, 3)
        #第二种：用在可迭代对象前，进行自动解包转化为多个单变量参数，例如：
        # def func(a, b, c):
        #     print(a, b, c)
        #
        # args = [1, 2, 3]
        # func(*args)  # 1 2 3
        #所以这里自动把这个列表解包了，并构成了Sequential结构
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        ret['hm']=self.hm(x)
        ret['wh'] = self.wh(x)
        ret['reg'] = self.reg(x)

        return [ret]

    #使用官方代码提供的方法：
    def init_weights(self, num_layers, pretrained=True):
        if pretrained:

            #初始化转置卷积和BN
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            #初始化一下三个输出层
            for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)

            #加载resnet预训练权重
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)

            print('\n=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)



def get_pose_net(num_layers, heads_classes,train_or_test='train'): #例如返回Bottleneck，[3, 4, 6, 3]
  layers = resnet_spec[num_layers]
  print ('\nresnet的层数是：',layers)#152
  model = PoseResNet(layers, heads_classes, head_conv=64)
  if train_or_test=='train':
    model.init_weights(num_layers)
  return model
