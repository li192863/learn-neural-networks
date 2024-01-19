import numpy as np
from PIL import Image


def cvtColor(image):
    """
    将图像转换成RGB图像，防止灰度图在预测时报错。
    :param image:
    :return:
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, letterbox_image):
    """
    对输入图像进行resize
    :param image:
    :param size: (width, height)
    :param letterbox_image: 是否对输入图像进行不失真的resize
    :return:
    """
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)  # 缩放比例
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)  # 宽高比不变等比例缩放
        new_image = Image.new('RGB', size, (128, 128, 128))  # 背景使用灰条填充
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 指定左上角坐标
    else:
        new_image = image.resize((w, h), Image.BICUBIC)  # 采样方式为PIL.Image.Resampling.BICUBIC
    return new_image


def preprocess_input(image):
    """
    预处理输入
    :param image:
    :return:
    """
    image /= 255.0
    return image


def get_classes(classes_path):
    """
    从文件中获取类别
    :param classes_path:
    :return:
    """
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchors(anchors_path):
    """
    从文件中获取先验框
    :param anchors_path:
    :return:
    """
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def show_config(**kwargs):
    """
    显示配置信息
    :param kwargs:
    :return:
    """
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


if __name__ == '__main__':
    class_names, num_classes = get_classes('../data/coco_classes.txt')
    print(class_names)
    print(num_classes)
    anchors, num_anchors = get_anchors('../data/yolo_anchors.txt')  # (array, 9)
    print(anchors)
    print(num_anchors)
