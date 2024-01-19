import colorsys
import os

import torch
from PIL import Image


class WSISegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, type, transforms=None, img_suffix='.jpg', seg_map_suffix='.png'):
        self.root = root
        self.type = type
        self.transforms = transforms
        self.classes = ['_background_', 'Normal', 'Tumor']

        # 加载所有图片文件，并对文件进行排序
        self.images_info_file = os.path.join(self.root, 'ImageSets', 'Segmentation', f'{self.type}.txt')
        self.images, self.labels = [], []
        with open(self.images_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                self.images.append(os.path.join(self.root, 'JPEGImages', f'{path}{img_suffix}'))
                self.labels.append(os.path.join(self.root, 'SegmentationClass', f'{path}{seg_map_suffix}'))

    def __getitem__(self, idx):
        # 加载图片以及蒙版
        # img_path = os.path.join(self.root, 'JPEGImages', self.images[idx])
        # mask_path = os.path.join(self.root, 'SegmentationClassPNG', self.labels[idx])
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.labels[idx]).convert('P', colors=len(self.classes))

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    classes = ['_background_', 'Normal', 'Tumor']
    num_classes = len(classes)
    print(len(classes))

    image = Image.open('../../datasets/Segmentation/JPEGImages/i0_j0_d5.0_TCGA-MR-A520-01Z-00-DX1.2F323BAC-56C9-4A0C-9C1B-2B4F776056B4.svs.jpg')
    image.show()
    mask1 = Image.open('../../datasets/Segmentation/SegmentationClass/i0_j0_d5.0_TCGA-MR-A520-01Z-00-DX1.2F323BAC-56C9-4A0C-9C1B-2B4F776056B4.svs.png').convert('P')
    mask1.show()
    import numpy as np
    mask1_arr = np.asarray(mask1)
    # 标签数字转化为标签名称
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]  # 色调 饱和度1 亮度1
    color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
    colors = []
    for c in color:
        colors.append(c[0])
        colors.append(c[1])
        colors.append(c[2])

    root = '../../datasets/Segmentation'
    dataset = WSISegmentationDataset(root, type='train')
    print(dataset[101])
    x, y = dataset[101]
    x.show()
    # y.convert('P')
    # y.putpalette(colors)
    import numpy as np

    y_arr = np.asarray(y)
    y.show()
