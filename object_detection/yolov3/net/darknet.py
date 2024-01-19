from collections import OrderedDict

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):  # C1xHxW
        identity = x

        out = self.conv1(x)  # C1xHxW
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # C1xHxW
        out = self.bn2(out)
        out = self.relu(out)

        out += identity  # C1xHxW
        return out  # C1xHxW


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # # 权值初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样
        layers.append(('ds_conv', nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(('ds_bn', nn.BatchNorm2d(planes[1])))
        layers.append(('ds_relu', nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append((f'residual_{i}', BasicBlock(self.inplanes, planes)))  # 不改变尺寸
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):  # 3x416x416
        x = self.conv1(x)  # 32x416x416
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 64x208x208
        x = self.layer2(x)  # 128x104x104
        out3 = self.layer3(x)  # 256x52x52
        out4 = self.layer4(out3)  # 512x26x26
        out5 = self.layer5(out4)  # 1024x13x13

        return out3, out4, out5  # (256x52x52, 512x26x26, 1024x13x13)


def DarkNet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model


if __name__ == '__main__':
    x = torch.randn((1, 3, 416, 416))
    model = DarkNet53()
    out3, out4, out5 = model(x)
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)
    print(model)
