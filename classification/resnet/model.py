import torch
from torch import nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Blocks for ResNet18/ResNet34
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道数的倍乘

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)  # stride != 1时降采样
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)  # activation
        self.downsample = downsample  # downsample

    def forward(self, x):  # C1xHxW
        identity = x

        out = self.conv1(x)  # C2xHxW (s=1) / C2x(H/2)x(W/2) (s=2)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # C2xHxW / C2x(H/2)x(W/2)
        out = self.bn2(out)

        if self.downsample is not None:  # 对应于虚线
            identity = self.downsample(x)  # None / C2x(H/2)x(W/2)

        out += identity  # C2xHxW / C2x(H/2)x(W/2)
        out = self.relu(out)

        return out  # C2xHxW (s=1) / C2x(H/2)x(W/2) (s=2)

# Blocks for ResNet50/ResNet101/ResNet152
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4  # 输出通道数的倍乘

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)  # stride != 1时降采样
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)  # activation
        self.downsample = downsample  # downsample

    def forward(self, x):  # C1xHxW
        identity = x

        out = self.conv1(x)  # C2xHxW
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # C2xHxW (s=1) / C2x(H/2)x(W/2) (s=2)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 4C2xHxW / 4C2x(H/2)x(W/2)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 4C2xHxW / 4C2x(H/2)x(W/2)
        out = self.relu(out)

        return out  # 4C2xHxW (s=1) / 4C2x(H/2)x(W/2) (s=2)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # stride=1
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        # 对于layer1，使用Bottleneck时，第一个Block（ResNet50/101/152）会进入if语句改变图片深度（而stride==1图片尺寸不变）
        # 对于layer2/3/4中第一个block均会进入if语句作降采样，剩余Block不做降采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 对应于虚线的降采样
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))  # 每个layer第一个Block
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # 每个layer剩余Block 均为实线
        return nn.Sequential(*layers)

    def forward(self, x):  # 3x224x224
        x = self.conv1(x)  # 64x112x112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64x56x56

        x = self.layer1(x)  # 64x56x56 (BasicBlock) / 256x56x56 (Bottleneck, layer1的第一个Block不改变尺寸)
        x = self.layer2(x)  # 128x28x28 / 512x28x28
        x = self.layer3(x)  # 256x14x14 / 1024x14x14
        x = self.layer4(x)  # 512x7x7 / 2048x7x7

        if self.include_top:
            x = self.avgpool(x) # 512x1x1 / 2048x1x1
            x = torch.flatten(x, 1)  # 512 / 2048
            x = self.fc(x)  # 1000 / 1000

        return x  # 1000

def ResNet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def ResNet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def ResNet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def ResNet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def ResNet152(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)

def NeuralNetwork(num_classes):
    """
    自定义神经网络 修改了ResNet最后全连接层输出维度
    :return:
    """
    model = ResNet18()

    in_channels = model.fc.in_features  # 获得最后fc层的in_features参数
    model.fc = nn.Linear(in_channels, num_classes)  # 改变原网络最后一层参数
    return model

if __name__ == "__main__":
    # model = ResNet18()
    # model = ResNet34()
    model = ResNet50()
    # model = ResNet101()
    # model = ResNet152()
    print(model)