# easy-pytorch-CenterNet
结构比较简洁的CenterNet

object as point那篇

pytorch新手，注释比较多

新冠期间在家无设备，代码为CPU版，无多线程，不需要build DCNv2，Windows也可以使用，特别适合极其低配机器，我家带不起英雄联盟的电脑都能跑

pytorch版本1.0.0

backbone只含resnet50~152

pycocotools windows按这个链接就好    http://www.mamicode.com/info-detail-2660241.html

官方代码地址  https://github.com/xingyizhou/CenterNet


# 使用方法
python main.py进行训练

训练完成后 python test.py进行测试

如需更改配置进入opts_and_utils.py
