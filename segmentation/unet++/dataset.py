import colorsys
import os

import torch
from PIL import Image


class FootballPlayerSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.classes = ['Goal Bar', 'Referee', 'Advertisement', 'Ground', 'Ball', 'Coaches & Officials', 'Audience',
                        'Goalkeeper A', 'Goalkeeper B', 'Team A', 'Team B']

        # 加载所有图片文件，并对文件进行排序
        self.files = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.imgs = [self.files[i] for i in range(len(self.files)) if i % 3 == 0]
        self.masks = [self.files[i] for i in range(len(self.files)) if i % 3 == 1]

    def __getitem__(self, idx):
        # 加载图片以及蒙版
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        mask_path = os.path.join(self.root, 'images', self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P', colors=len(self.classes))

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    classes = ['Goal Bar', 'Referee', 'Advertisement', 'Ground', 'Ball', 'Coaches & Officials', 'Audience',
                        'Goalkeeper A', 'Goalkeeper B', 'Team A', 'Team B']
    num_classes = len(classes)
    print(len(classes))

    # image = Image.open('../../datasets/Motorcycle Night Ride/images/night ride (7).png')
    # image.show()
    # mask1 = Image.open('../../datasets/Motorcycle Night Ride/images/night ride (7).png___fuse.png').convert('P')
    # mask1.show()
    # import numpy as np
    # mask1_arr = np.asarray(mask1)
    # # 标签数字转化为标签名称
    # hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]  # 色调 饱和度1 亮度1
    # color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
    # colors = []
    # for c in color:
    #     colors.append(c[0])
    #     colors.append(c[1])
    #     colors.append(c[2])

    root = '../../datasets/Football Player Segmentation'
    dataset = FootballPlayerSegmentationDataset(root)
    print(dataset[1])
    x, y = dataset[1]
    x.show()
    # y.convert('P')
    # y.putpalette(colors)
    import numpy as np

    y_arr = np.asarray(y)
    y.show()
