from collections import OrderedDict

import torch
from torch import nn

from darknet import DarkNet53


def conv2d(in_channels, out_channels, kernel_size):
    """
    创建卷积层 C1xHxW -> C2xHxW
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :return:
    """
    padding = (kernel_size - 1) // 2
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.LeakyReLU(0.1))
    ]))


def make_last_layers(filters_list, in_channels, out_channels):
    """
    创建末层 C1xHxW -> C2xHxW
    :param filters_list:
    :param in_channels:
    :param out_channels:
    :return:
    """
    return nn.Sequential(
        conv2d(in_channels, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),

        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    )


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes):
        super(YoloBody, self).__init__()
        self.backbone = DarkNet53()
        out_filters = self.backbone.layers_out_filters  # [64, 128, 256, 512, 1024]

        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (5 + num_classes))

        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (5 + num_classes))

        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (5 + num_classes))

    def forward(self, x):  # 3x416x416
        x2, x1, x0 = self.backbone(x)  # (256x52x52, 512x26x26, 1024x13x13)

        out0_branch = self.last_layer0[0:5](x0)  # 512x13x13 (x0)
        out0 = self.last_layer0[5:7](out0_branch)  # 255x13x13 (x0)

        x1_in = self.last_layer1_conv(out0_branch)  # 256x13x13 (x0)
        x1_in = self.last_layer1_upsample(x1_in)  # 256x26x26 (x0)
        x1_in = torch.cat([x1_in, x1], dim=1)  # 784x26x26 (x0, x1)
        out1_branch = self.last_layer1[0:5](x1_in)  # 256x26x26 (x0, x1)
        out1 = self.last_layer1[5:7](out1_branch)  # 255x26x26 (x0, x1)

        x2_in = self.last_layer2_conv(out1_branch)  # 128x26x26 (x0, x1)
        x2_in = self.last_layer2_upsample(x2_in)  # 128x52x52 (x0, x1)
        x2_in = torch.cat([x2_in, x2], dim=1)  # 384x52x52 (x0, x1, x2)
        out2 = self.last_layer2(x2_in)  # 255x52x52 (x0, x1, x2)

        return out0, out1, out2  # (255x13x13, 255x26x26, 255x52x52)


if __name__ == '__main__':
    x = torch.randn((1, 3, 416, 416))
    anchors = [[[10, 13], [16, 30], [33, 23]],
               [[30, 61], [62, 45], [59, 119]],
               [[116, 90], [156, 198], [373, 326]]]
    model = YoloBody(anchors, 80)
    out_large, out_medium, out_small = model(x)
    print(out_large.shape)
    print(out_medium.shape)
    print(out_small.shape)
    print(model)
