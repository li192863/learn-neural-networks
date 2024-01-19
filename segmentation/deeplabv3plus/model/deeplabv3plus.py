from functools import partial
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from .mobilenetv3 import mobilenet_v3_large


class MobileNetV3(nn.Module):
    """ backbone基于mobilenet修改而来 """

    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV3, self).__init__()
        # backbone
        model = mobilenet_v3_large(pretrained)

        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 13]  # 网络在这些层发生了降采样

        # 降采样倍数
        if downsample_factor == 8:
            # 进行三次降采样
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))  # 对某个模块递归进行修改
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            # 进行四次降采样
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        """ stride修改为1，并将非降采样层的中间卷积层变为空洞卷积 """
        classname = m.__class__.__name__
        if classname.find('block') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                # o = (i + 2 * p - d * (k - 1) - 1) // s + 1
                # d * (k - 1) = 2 * p
                d, k = dilate // 2, m.kernel_size[0]
                p = d * (k - 1) // 2
                m.dilation = (d, d)
                m.padding = (p, p)
            else:
                # o = (i + 2 * p - d * (k - 1) - 1) // s + 1
                # d * (k - 1) = 2 * p
                d, k = dilate, m.kernel_size[0]
                p = d * (k - 1) // 2
                m.dilation = (d, d)
                m.padding = (p, p)

    def forward(self, x):
        low_level_features = self.features[:4](x)  # 浅层语义信息
        x = self.features[4:](low_level_features)  # 深层语义信息
        return low_level_features, x


class ASPPConv(nn.Sequential):
    """ ASPP卷积分支 """

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    """ ASPP池化分支 """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """ ASPP模块 """

    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        # 分支1
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        )
        # 分支2/3/4
        rates = tuple(atrous_rates)  # [6, 12, 18]
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        # 分支5
        modules.append(ASPPPooling(in_channels, out_channels))
        # 合并所有分支
        self.convs = nn.ModuleList(modules)
        # 1x1卷积压缩
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """ DeepLabV3+模型 """

    def __init__(self, num_classes, pretrained=True, downsample_factor=8):
        super(DeepLabV3Plus, self).__init__()
        # 获得两个特征层
        # 浅层特征    [128,128,24]
        # 主干部分    [16,16,160]
        self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
        in_channels = 160
        low_level_channels = 24

        # ASPP特征提取模块
        self.aspp = ASPP(in_channels=in_channels, atrous_rates=[12, 24, 32], out_channels=256)

        # 浅层特征边
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # 获得两个特征层
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # 将加强特征边上采样，与浅层特征堆叠后利用卷积进行特征提取
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        # 分类
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    model = DeepLabV3Plus(7)
    print(model)
    x = torch.randn((2, 3, 512, 512))
    print(model(x).shape)
