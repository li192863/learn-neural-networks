import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import os.path
import sys

from util.util import cvtColor, preprocess_input

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

DS_CLASSES = ('with_mask', 'without_mask', 'mask_weared_incorrect')  # 数据集类型
INPUT_SHAPE = [416, 416]  # height, width

class BoundingBox(object):
    """
    边界框类，可实现常见边界框形式的相互转化。常见格式有
        'voc':  [xmin, ymin, xmax, ymax]
        'coco': [xmin, ymin, width, height]
        'yolo': [x_center, y_center, width_ratio, height_ratio]
    """
    format_types = ['voc', 'coco', 'yolo']

    def __init__(self, boxes, format, dtype=np.float64) -> None:
        """
        构造边界框对象
        :param boxes: 边界框
        :param format: 边界框数据本身格式('voc'/'coco'/'yolo')
        :param dtype: 存储的格式
        """
        if format.lower() not in self.format_types:
            raise ValueError(f'Unsupported format \'{format}\' !')
        self.boxes = np.array(boxes, dtype=dtype).reshape(-1, 4)
        self.format = format.lower()
        self.dtype = dtype

    def get_boxes(self, format=None, image_width=None, image_height=None):
        """
        获取边界框
        :param format: 边界框格式('voc'/'coco'/'yolo')
        :param image_width: 图片宽
        :param image_height: 图片高
        :return: 边界框
        """
        format = format or self.format
        # 判断是否需要转化
        if format != self.format:
            self.convert(format, image_width, image_height)
        # 根据格式进一步转化
        if format == 'voc' or format == 'coco':
            return np.int32(self.boxes)
        elif format == 'yolo':
            return np.float64(self.boxes)
        else:
            raise ValueError('Internal format error!')

    def convert(self, desired_format: str, image_width=None, image_height=None):
        """
        转换边界框为指定格式
        :param desired_format: 需要的边界框格式('voc'/'coco'/'yolo')
        :param image_width: 图片宽
        :param image_height: 图片高
        """
        if desired_format not in self.format_types:
            raise ValueError('Desired type is unsupported!')
        if desired_format == 'yolo' and (image_width == None or image_height == None):
            raise ValueError('Image width and height must be provided to convert to Yolo format!')
        if self.format == 'voc':
            if desired_format == 'yolo':
                self.boxes = self.voc2yolo(image_width, image_height)
            elif desired_format == 'coco':
                self.boxes = self.voc2coco()
            else:
                pass
        elif self.format == 'coco':
            if desired_format == 'yolo':
                self.boxes = self.coco2yolo(image_width, image_height)
            elif desired_format == 'voc':
                self.boxes = self.coco2voc()
            else:
                pass
        elif self.format == 'yolo':
            if desired_format == 'voc':
                self.boxes = self.yolo2voc(image_width, image_height)
            elif desired_format == 'coco':
                self.boxes = self.yolo2coco(image_width, image_height)
            else:
                pass
        else:
            raise ValueError('Internal format error!')

    def voc2yolo(self, image_width, image_height):
        """ voc格式转为yolo格式 """
        self.format = 'yolo'
        # [xmin, ymin, xmax, ymax] -> [x_center, y_center, width_ratio, height_ratio]
        x_center = (self.boxes[:, 0] + self.boxes[:, 2]) / 2.0 / image_width
        y_center = (self.boxes[:, 1] + self.boxes[:, 3]) / 2.0 / image_height
        width_ratio = (self.boxes[:, 2] - self.boxes[:, 0]) / image_width
        height_ratio = (self.boxes[:, 3] - self.boxes[:, 1]) / image_height
        return np.stack((x_center, y_center, width_ratio, height_ratio), axis=-1)

    def voc2coco(self):
        """ voc格式转为coco格式 """
        self.format = 'coco'
        # [xmin, ymin, xmax, ymax] -> [xmin, ymin, width, height]
        xmin = np.floor(self.boxes[:, 0])
        ymin = np.floor(self.boxes[:, 1])
        width = np.floor(self.boxes[:, 2] - self.boxes[:, 0])  # 数字差异较大时可能精度不够，使用floor确保其为整
        height = np.floor(self.boxes[:, 3] - self.boxes[:, 1])
        return np.stack((xmin, ymin, width, height), axis=-1)

    def coco2yolo(self, image_width, image_height):
        """ coco格式转为yolo格式 """
        self.format = 'yolo'
        # [xmin, ymin, width, height] -> [x_center, y_center, width_ratio, height_ratio]
        x_center = (2 * self.boxes[:, 0] + self.boxes[:, 2]) / 2.0 / image_width
        y_center = (2 * self.boxes[:, 1] + self.boxes[:, 3]) / 2.0 / image_height
        width_ratio = self.boxes[:, 2] / image_width
        height_ratio = self.boxes[:, 3] / image_height
        return np.stack((x_center, y_center, width_ratio, height_ratio), axis=-1)

    def coco2voc(self):
        """ coco格式转为voc格式 """
        self.format = 'voc'
        # [xmin, ymin, width, height] -> [xmin, ymin, xmax, ymax]
        xmin = np.floor(self.boxes[:, 0])
        ymin = np.floor(self.boxes[:, 1])
        xmax = np.floor(self.boxes[:, 2] + self.boxes[:, 0])  # 数字差异较大时可能精度不够，使用floor确保其为整
        ymax = np.floor(self.boxes[:, 3] + self.boxes[:, 1])
        return np.stack((xmin, ymin, xmax, ymax), axis=-1)

    def yolo2voc(self, image_width, image_height):
        """ yolo格式转为voc格式 """
        self.format = 'voc'
        # [x_center, y_center, width_ratio, height_ratio] -> [xmin, ymin, xmax, ymax]
        box_width = self.boxes[:, 2] * image_width
        box_height = self.boxes[:, 3] * image_height
        xmin = np.floor((2 * self.boxes[:, 0] * image_width - box_width) / 2.0)
        ymin = np.floor((2 * self.boxes[:, 1] * image_height - box_height) / 2.0)
        xmax = np.floor(xmin + box_width)
        ymax = np.floor(ymin + box_height)
        return np.stack((xmin, ymin, xmax, ymax), axis=-1)

    def yolo2coco(self, image_width, image_height):
        """ yolo格式转为coco格式 """
        self.format = 'coco'
        # [x_center, y_center, width_ratio, height_ratio] -> [xmin, ymin, width, height]
        box_width = self.boxes[:, 2] * image_width
        box_height = self.boxes[:, 3] * image_height
        xmin = np.floor((2 * self.boxes[:, 0] * image_width - box_width) / 2.0)
        ymin = np.floor((2 * self.boxes[:, 1] * image_height - box_height) / 2.0)
        width = np.floor(box_width)
        height = np.floor(box_height)
        return np.stack((xmin, ymin, width, height), axis=-1)


class VOCTransform(object):
    """
    转换VOC注解类。默认情况下，使用全局常量定义的类名(class_to_idx)，跳过困难照片(keep_difficult)，默认生成的标签文件为voc格式(format)，
    标签类名下标为0(class_label_pos)，读取的xml文件路径为 'annotations' (xml_path)，标签存放位置为 'labels' (txt_path)
    """

    def __init__(self, root, class_to_idx=None, class_label_pos=0, keep_difficult=False):
        """
        构造转换注解对象
        :param root: 数据集根目录
        :param class_to_idx: 字典 {类名0:0, 类名1: 1, ...}
        :param class_label_pos: 类名下标放置位置
        :param keep_difficult: 是否保留困难值
        """
        self.root = root
        self.class_to_idx = class_to_idx or dict(zip(DS_CLASSES, range(len(DS_CLASSES))))
        self.class_label_pos = class_label_pos
        self.keep_difficult = keep_difficult

    def __call__(self, target, format='voc'):
        """
        转换注解生成列表
        :param target: xml文件根节点
        :param format: 格式
        :return: 列表
        """
        res = []
        for obj in target.iter('object'):
            # 跳过困难项
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            # 获取名称
            name = obj[0].text.lower().strip()
            # 获取宽高
            width = int(target.find('size').find('width').text)
            height = int(target.find('size').find('height').text)
            # 获取边界框[xmin, ymin, xmax, ymax]
            bndbox = [int(bb.text) - 1 for bb in obj[5]]
            # 转换格式
            box = BoundingBox(bndbox, format='voc').get_boxes(format, width, height)
            box = box.tolist()[0]
            # 添加类号
            class_idx = self.class_to_idx[name]
            box.insert(self.class_label_pos, class_idx)

            res += [box]
        return res

    def get_txt(self, target, format='voc'):
        """
        生成指定文本
        :param target: xml文件根节点
        :param format: 格式
        :return: xml对应的文本信息
        """
        boxes = self(target, format=format)
        res = ''
        for box in boxes:
            line = ''
            for item in box:
                line += str(item) + ' '
            res += line.strip() + '\n'
        return res

    def transform_to_txt(self, format='voc', xml_path='annotations', txt_path='labels'):
        """
        转换xml为txt文本
        :param format: 格式
        :param xml_path: 存放xml文件夹路径
        :param txt_path: 存放txt文件夹路径
        """
        xml_path = os.path.join(self.root, xml_path)
        txt_path = os.path.join(self.root, txt_path)

        if not os.path.exists(txt_path):  # 若目录不存在则创建
            os.makedirs(txt_path)
        # 遍历存放xml的文件夹
        for dirname, _, filenames in os.walk(xml_path):
            for filename in filenames:
                target = ET.parse(os.path.join(dirname, filename)).getroot()
                txt_string = self.get_txt(target, format)
                with open(os.path.join(txt_path, filename.split('.')[0] + '.txt'), 'w+') as f:
                    f.write(txt_string)


class YoloDataset(Dataset):
    """
    Yolo数据集类。默认情况下，存放划分后文本信息的文件夹为datasets(datasets_dir)，图片及标签默认配置信息见images_conf和labels_conf
    """

    def __init__(self, root, type, datasets_dir='datasets', images_conf=None, labels_conf=None, transform=None,
                 target_transform=None):
        """
        构造数据集对象
        :param root: 数据集根目录
        :param type: 获取的数据集类型（train/val/test）
        :param datasets_dir: 存放划分后训练/测试/验证信息文件夹
        :param images_conf: 图片配置
        :param labels_conf: 标签配置
        :param transform: 图片转换
        :param target_transform: 标签转换
        """
        self.root = root  # 根目录
        self.type = type  # 类型，含train/val/test
        self.images_conf = images_conf or {'images_dir': 'images', 'default_shape': [416, 416]}  # 图片配置
        self.labels_conf = labels_conf or {'labels_dir': 'labels', 'labels_format': 'voc', 'class_label_pos': 0}  # 标签格式
        self.transform = transform
        self.target_transform = target_transform

        self.datasets_dir = datasets_dir  # 划分后训练/测试/验证信息文件夹
        self._dataset_path = os.path.join(self.root, datasets_dir, '%s.txt')  # 获取划分的文本文件
        self._read_dataset_file()  # 获取划分后的所有图片id（即图片名称）

        self.images_dir = self.images_conf['images_dir']  # 图片文件夹
        self._image_path = os.path.join(self.root, self.images_dir, f'%s.{self._image_format}')

        self.labels_dir = self.labels_conf['labels_dir']  # 标签文件夹
        self._label_path = os.path.join(self.root, self.labels_dir, '%s.txt')

    def _read_dataset_file(self):
        """ 获取数据集图片信息 """
        with open(self._dataset_path % self.type) as f:
            lines = f.readlines()
        # 图片名称列表
        self.ids = [os.path.split(line.strip())[-1].split('.')[0] for line in lines]
        # 图片后缀名
        self._image_format = os.path.split(lines[0].strip())[-1].split('.')[-1]  # 图片格式(以第一张图片的格式为准)
        # 图片数量
        self.length = len(self.ids)

    def __len__(self):
        """ 获取数据集大小 """
        return self.length

    def __getitem__(self, index, preprocess=True):
        """ 获取数据集元素 """
        # 获取当前索引对应的图片名称
        index = index % self.length
        id = self.ids[index]

        # 获取图片
        image = Image.open(self._image_path % id)
        image_height, image_width, _ = np.shape(image)
        # 获取标签
        with open(self._label_path % id) as f:
            lines = f.readlines()
        labels = [line.strip() for line in lines]
        target = []
        for label in labels:
            target.append([eval(val_str) for val_str in label.split()])
        target = np.array(target)

        # 预处理
        if preprocess:
            # 图片预处理
            image = cvtColor(image)  # 转为3通道RGB图像
            image = image.resize([self.images_conf['default_shape'][1], self.images_conf['default_shape'][0]], Image.BICUBIC)  # 缩放至指定大小
            image = preprocess_input(np.array(image, dtype=np.float32))  # 归一化
            image = np.transpose(image, (2, 0, 1))  # 深度维度移至最前

            # 标签预处理
            box_pos = [0, 1, 2, 3, 4]  # 定义边界框的所有列
            box_pos.remove(self.labels_conf['class_label_pos'])  # 删除类别对应的列号
            boxes = BoundingBox(target[:, box_pos], format=self.labels_conf['labels_format']).get_boxes('yolo',
                                                                                                        image_width,
                                                                                                        image_height)  # 转换为yolo格式
            target = np.float32(np.concatenate((boxes, target[:, self.labels_conf['class_label_pos']].reshape(-1, 1)),
                                    axis=-1))  # 合并标签信息
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def show(self, index, preprocess=False):
        """
        显示第index张图片
        :param index: 下标
        :param transformed: 是否显示经过预处理的图片
        :return:
        """
        index = index % self.length
        image, target = self.__getitem__(index, preprocess)

        box_pos = [0, 1, 2, 3, 4]  # 定义边界框的所有列
        box_pos.remove(self.labels_conf['class_label_pos'])  # 删除类别对应的列号
        # 若显示经过预处理的照片则需进行相应解码
        if preprocess:
            image = Image.fromarray(np.uint8(np.transpose(image, (1, 2, 0)) * 255))
            image_height, image_width, _ = np.shape(image)  # 获取当前图片宽高

            boxes = BoundingBox(target[:, box_pos], format='yolo').get_boxes('voc', image_width,
                                                                             image_height)  # 转换为voc格式
            target = np.concatenate((boxes, target[:, self.labels_conf['class_label_pos']].reshape(-1, 1)),
                                    axis=-1)  # 合并标签信息

        draw = ImageDraw.Draw(image)
        for obj in target:
            draw.rectangle(((obj[box_pos[0]], obj[box_pos[1]]), (obj[box_pos[2]], obj[box_pos[3]])),
                           outline=(255, 0, 0))
            draw.text((obj[box_pos[0]], obj[box_pos[1]]), str(obj[self.labels_conf['class_label_pos']]),
                      fill=(0, 255, 0))
        image.show()


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


if __name__ == '__main__':
    # 测试边界框
    # box = np.abs(np.clip(np.random.rand(5, 4), 0, 1))
    print('--------------------------')
    box = np.abs(np.random.rand(5, 4) * 450)
    bbox = BoundingBox(box, 'coco')
    print(bbox.boxes)
    bbox.convert('voc', 400, 500)  # format, width, height
    print(bbox.boxes)
    bbox.convert('coco', 400, 500)  # format, width, height
    print(bbox.boxes)

    # 生成所有文件的标签
    print('--------------------------')
    ROOT = r'E:\Projects\#Project_Python\datasets\face-mask-detection'
    transform = VOCTransform(ROOT, class_label_pos=4)  # 指定类名放置最后
    transform.transform_to_txt()  # 格式默认为voc

    # 创建数据集
    print('--------------------------')
    images_conf = {'images_dir': 'images', 'default_shape': INPUT_SHAPE}  # 图片配置
    labels_conf = {'labels_dir': 'labels', 'labels_format': 'voc', 'class_label_pos': 4}  # 标签格式
    ds = YoloDataset(ROOT, 'train', images_conf=images_conf, labels_conf=labels_conf)

    # 取出训练集的一张图片查看效果
    print(len(ds))
    image, target = ds[1]
    print(image.shape)
    print(target.shape)
    ds.show(512, False)
    ds.show(512, True)
