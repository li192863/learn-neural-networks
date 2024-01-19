import argparse
import random
import torch

from yolo import Yolo

from dataset import YoloDataset

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_SHAPE = [416, 416]


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    ROOT = r'E:\Projects\#Project_Python\datasets\face-mask-detection'  # 数据集根目录
    images_conf = {'images_dir': 'images', 'default_shape': INPUT_SHAPE}  # 图片配置
    labels_conf = {'labels_dir': 'labels', 'labels_format': 'voc', 'class_label_pos': 4}  # 标签格式

    test_data = YoloDataset(root=ROOT, type='test', images_conf=images_conf, labels_conf=labels_conf)
    x, y = test_data.__getitem__(random.randint(0, len(test_data) - 1), preprocess=False)  # 随机获取测试数据
    return x, y


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()


def main(opt):
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    x, y = get_test_data(opt)
    # 模型
    model = Yolo()
    # 评估
    with torch.no_grad():  # Disabling gradient calculation
        image = model.detect_image(x)
        image.show()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
