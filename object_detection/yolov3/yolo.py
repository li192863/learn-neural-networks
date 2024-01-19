import colorsys
import os

import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from net.body import YoloBody
from util.util import get_classes, get_anchors, show_config, cvtColor, resize_image, preprocess_input
from util.util_bbox import DecodeBox


class Yolo(object):
    _defaults = {
        # 权重/类别
        'model_path': 'data/fmd_model.pth',
        'classes_path': 'data/FMD_classes.txt',
        # 先验框
        'anchors_path': 'data/yolo_anchors.txt',
        'anchors_mask': [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # 输入图片大小(height, width)
        'input_shape': [416, 416],
        # 置信度阈值
        'confidence': 0.01,  # 只有得分大于置信度的预测框会被保留下来
        # 非极大值抑制的nms_iou
        'nms_iou': 0.3,
        # 是否对输入图像进行不失真（宽高比不变）的resize
        'letterbox_image': False,
        # 使用设备
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    @classmethod
    def get_defaults(cls, name):
        if name in cls._defaults:
            return cls._defaults[name]
        else:
            raise NameError(f'Unrecognized attribute name \'{name}\'.')

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # 初始化_defaults
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        # 获得种类和先验框
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.device,
                                   self.anchors_mask)
        # 画框设置不同的颜
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]  # 色调 饱和度1 亮度1
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 生成模型
        self.generate()

        # 显示配置
        show_config(**self._defaults)

    def generate(self):
        """
        生成模型
        :return:
        """
        self.net = YoloBody(self.anchors_mask, self.num_classes).to(self.device)
        # 加载权重
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print(f'{self.model_path} model, anchors, and classes loaded.')

    def detect_image(self, image, crop=False, count=False):
        """
        检测图片
        :param image:
        :param crop:
        :param count:
        :return:
        """
        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)  # 通道数改为3
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)  # 缩放图片大小
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)),
                                    0)  # 归一化为4D tensor(1, 3, width, height)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.device)

            # 将图像输入网络当中进行预测
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(
                outputs)  # (batch_size, num_anchors * input_width * input_height, 4 + 1 + num_classes)

            # 将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_threshold=self.confidence,
                                                         nms_threshold=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')  # 类别，以数字表示
            top_conf = results[0][:, 4] * results[0][:, 5]  # 置信度 * 种类置信度
            top_boxes = results[0][:, :4]  # 边框线，(y1, x1, y2, x2)

        # 设置字体与边框厚度
        font = ImageFont.truetype(font='data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # 计数
        if count:
            print('top_label', top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        # 剪裁
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]  # 边框线，(y1, x1, y2, x2)
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # 图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box  # 边框线，(y1, x1, y2, x2)

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])  # 文本左上角点
            else:  # 边界框label超出图片
                text_origin = np.array([left, top + 1])  # 文本左上角点

            for i in range(thickness):
                draw.rectangle((left + i, top + i, right - i, bottom - i), outline=self.colors[c])  # 边界框
            draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)), fill=self.colors[c])  # 文本
            draw.text(tuple(text_origin), str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


if __name__ == '__main__':
    yolo = Yolo(test=123)
    print(yolo.get_defaults('test'))
    image = yolo.detect_image(Image.fromarray(np.random.randn(416, 416) * 255))
    image.show()
