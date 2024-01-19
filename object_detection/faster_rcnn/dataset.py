import os
import torch
import torch.utils.data

from PIL import Image


class FaceMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 加载所有图片文件，并对文件进行排序
        self.images = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.labels = list(sorted(os.listdir(os.path.join(root, 'labels'))))

    def __getitem__(self, idx):
        # 加载图片以及标签
        img_path = os.path.join(self.root, 'images', self.images[idx])
        label_path = os.path.join(self.root, 'labels', self.labels[idx])
        img = Image.open(img_path).convert('RGB')
        label = open(label_path)  # 注意此时label为文件，包含了边界框信息以及类别信息

        # 获取标签
        label_str_list = [line.strip() for line in label.readlines()]
        boxes, labels, num_objs = [], [], 0
        for label_str in label_str_list:
            # 获取label信息
            xmin, ymin, xmax, ymax, class_idx = [eval(val_str) for val_str in label_str.split()]

            boxes.append([xmin, ymin, xmax, ymax])  # 各个物体的边界框
            labels.append(class_idx + 1)  # 各个物体的类别（注意此时类别需要加1，因为标签文件中class_idx从0开始，而fast_rcnn中0为背景）
            num_objs += 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)  # 假定所有实例都不拥挤

        # 获取目标
        target = {}
        target['boxes'] = boxes  # FloatTensor[N, 4]
        target['labels'] = labels  # Int64Tensor[N]
        target['image_id'] = image_id  # Int64Tensor[1]
        target['area'] = area  # Tensor[N]
        target['iscrowd'] = iscrowd  # UInt8Tensor[N]
        # target['masks'] = masks  # 可选项 UInt8Tensor[N, H, W] The segmentation masks for each one of the objects
        # target['keypoints'] = keypoints  # 可选项 FloatTensor[N, K, 3] 3 for [x, y, visibility] format

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    root = '../../datasets/face-mask-detection'
    dataset = FaceMaskDataset(root)
    print(dataset[0])
