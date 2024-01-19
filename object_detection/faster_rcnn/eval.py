import argparse
import colorsys
import math

import torch
import torchvision.transforms.functional as F
from PIL import ImageOps
from torchvision.utils import draw_bounding_boxes, make_grid

import utils as utils
from dataset import FaceMaskDataset
from model import get_model_object_detection
from presets import DetectionPresetEval

DATASET_ROOT_PATH = '../../../datasets/face-mask-detection'
DEFAULT_MODEL_PATH = 'data/model.pth'
DEFAULT_BATCH_SIZE = 6
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_WORKERS = 16
classes = ['background', 'with_mask', 'without_mask', 'mask_weared_incorrect']


def get_test_data(opt):
    """
    获取测试数据
    :return:
    """
    test_data = FaceMaskDataset(DATASET_ROOT_PATH, DetectionPresetEval())
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=True,
                                                  num_workers=opt.workers, collate_fn=utils.collate_fn)

    x, y = next(iter(test_dataloader))
    x = list(image.to(opt.device) for image in x)
    y = [{k: v.to(opt.device) for k, v in t.items()} for t in y]
    return x, y


def show_detection_result(images, boxes, labels, image_size=None, colors=None):
    """
    展示目标检测结果
    :param images:
    :param boxes:
    :param labels:
    :param image_size:
    :param colors:
    :return:
    """
    # 预处理图片
    images = [torch.as_tensor(image.cpu() * 255, dtype=torch.uint8) for image in images]  # 对输入tensor进行处理

    # 绘制每张图
    num_images = len(images)
    image_size = image_size or (sum(list(map(lambda x: x.shape[2], images))) // num_images,
                                sum(list(map(lambda x: x.shape[1], images))) // num_images)  # 获取图片大小(W, H)
    for i in range(num_images):
        image = draw_bounding_boxes(images[i], boxes[i], labels[i], colors[i])
        image = F.to_pil_image(image)

        image = letterbox_image(image, image_size)

        image = F.to_tensor(image)
        images[i] = image

    # 生成网格图
    # nrow = 8
    nrow = int(math.ceil(math.sqrt(len(images))))
    result = make_grid(images, nrow=nrow)
    result = F.to_pil_image(result)
    result.save('data/result.png')
    result.show()


def letterbox_image(image, image_size):
    """
    图片等比例缩放
    :param image: PIL image
    :param image_shape: (W, H)
    :return:
    """
    # 获取原始宽高和需要的宽高
    old_width, old_height = image.size
    new_width, new_height = image_size
    # 缩放图片有效区域
    scale = min(new_width / old_width, new_height / old_height)  # 图片有效区域缩放比例
    valid_width, valid_height = int(old_width * scale), int(old_height * scale)
    image = image.resize((valid_width, valid_height))
    # 填充图片无效区域
    pad_color = (128, 128, 128)
    result = ImageOps.pad(image, (new_width, new_height), color=pad_color)
    return result


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model weights path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', default=DEFAULT_WORKERS, help='max dataloader workers')
    return parser.parse_args()


def main(opt):
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    x, y = get_test_data(opt)
    # 模型
    num_classes = len(classes)
    model = get_model_object_detection(num_classes).to(opt.device)
    # 参数
    model.load_state_dict(torch.load(opt.model_path))
    # 评估
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        pred = model(x)

        # 按照分数过滤
        score_threshold = 0.75
        mask = [out['scores'] > score_threshold for out in pred]  # 获取蒙版
        scores = [out['scores'][mask] for out, mask in zip(pred, mask)]  # 获取分数
        boxes = [out['boxes'][mask] for out, mask in zip(pred, mask)]  # 获取边界框
        labels = [out['labels'][mask] for out, mask in zip(pred, mask)]  # 获取标签

        # 标签数字转化为标签名称
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]  # 色调 饱和度1 亮度1
        color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
        colors = []
        for i, label in enumerate(labels):
            labels[i] = [classes[i] for i in label]
            colors.append([color[i] for i in label])
        show_detection_result(x, boxes, labels, image_size=[640, 640], colors=colors)
        print(labels)
        print(scores)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
