import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from unet import UNet


def get_model_sematic_segmentation(num_classes):
    model = UNet(3, num_classes)

    return model

if __name__ == '__main__':
    model = get_model_sematic_segmentation(6)

    print(model)