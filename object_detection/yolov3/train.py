import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import YoloLoss
from dataset import YoloDataset, yolo_dataset_collate
from yolo import YoloBody
from util.util import get_classes, get_anchors

INPUT_SHAPE = [416, 416]  # height, width
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'data/fmd_model.pth'
DEFAULT_MODEL_PATH = 'data/yolo_weights.pth'
DEFAULT_CLASSES_PATH = 'data/FMD_classes.txt'
DEFAULT_ANCHORS_PATH = 'data/yolo_anchors.txt'


def get_dataloader(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    ROOT = r'E:\Projects\#Project_Python\datasets\face-mask-detection'  # 数据集根目录
    images_conf = {'images_dir': 'images', 'default_shape': INPUT_SHAPE}  # 图片配置
    labels_conf = {'labels_dir': 'labels', 'labels_format': 'voc', 'class_label_pos': 4}  # 标签格式

    training_data = YoloDataset(root=ROOT, type='train', images_conf=images_conf, labels_conf=labels_conf)
    test_data = YoloDataset(root=ROOT, type='val', images_conf=images_conf, labels_conf=labels_conf)

    train_dataloader = DataLoader(training_data, batch_size=opt.batch_size, num_workers=4,
                                  collate_fn=yolo_dataset_collate)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=4, collate_fn=yolo_dataset_collate)
    return train_dataloader, test_dataloader


def get_model(opt):
    """
    获取模型
    :param opt:
    :return:
    """
    # 数据
    class_names, num_classes = get_classes(opt.classes_path)
    # 模型
    model = YoloBody(ANCHORS_MASK, num_classes).to(opt.device)
    # 权重
    model_dict = model.state_dict()
    pretrained_dict = torch.load(opt.model_path, map_location=opt.device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def get_loss(opt):
    """
    获取损失函数
    :param opt:
    :return:
    """
    # 数据
    class_names, num_classes = get_classes(opt.classes_path)
    anchors, num_anchors = get_anchors(opt.anchors_path)
    # 参数
    yolo_loss = YoloLoss(anchors, num_classes, INPUT_SHAPE, opt.device, ANCHORS_MASK)  # 损失函数
    def loss_fn(pred, targets):
        loss_value_all = 0
        num_pos_all = 0
        for l in range(len(pred)):
            loss_item, num_pos = yolo_loss(l, pred[l], targets)
            loss_value_all += loss_item
            num_pos_all += num_pos
        loss_value = loss_value_all / num_pos_all
        return loss_value
    return loss_fn


def train(dataloader, model, loss_fn, optimizer, opt):
    """
    训练模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param opt:
    :return:
    """
    model.train()  # Sets the module in training mode
    with tqdm(dataloader, desc=f'Epoch {opt.epoch}/{opt.epochs}, train', total=len(dataloader)) as pbar:  # 进度条
        for X, y in pbar:
            # 数据处理
            X = torch.FloatTensor(torch.from_numpy(X)).to(opt.device)  # Tensor (batch_size, 3, input_shape[1], input_shape[0])
            y = [torch.FloatTensor(torch.from_numpy(ann)).to(opt.device) for ann in y] # list 元素为(num_targets, 5)

            # 前向传播
            pred = model(X)  # 预测结果
            loss = loss_fn(pred, y)  # 计算损失

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            # 打印信息
            pbar.set_postfix({'loss': f'{loss.item():>7f}'})


def test(dataloader, model, loss_fn, opt):
    """
    测试模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param opt:
    :return:
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()  # Sets the module in evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        with tqdm(dataloader, desc=' ' * (len(str(opt.epoch)) + len(str(opt.epochs)) + 9) + 'test',
                  total=len(dataloader)) as pbar:  # 进度条
            for X, y in pbar:
                # 数据处理
                X = torch.FloatTensor(torch.from_numpy(X)).to(
                    opt.device)  # Tensor (batch_size, 3, input_shape[1], input_shape[0])
                y = [torch.FloatTensor(torch.from_numpy(ann)).to(opt.device) for ann in y]  # list 元素为(num_targets, 5)

                # 前向传播
                pred = model(X)  # 预测结果
                test_loss += loss_fn(pred, y).item()  # 计算损失
                pbar.set_postfix({'Avg loss': f'{test_loss / num_batches:>8f}'})


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-path', default=DEFAULT_SAVE_PATH, help='model save path')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='pretrained model path')
    parser.add_argument('--classes-path', default=DEFAULT_CLASSES_PATH, help='classes path')
    parser.add_argument('--anchors-path', default=DEFAULT_ANCHORS_PATH, help='anchors path')
    return parser.parse_args()


def main(opt):
    """
    主函数
    :param opt:
    :return:
    """
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    train_dataloader, test_dataloader = get_dataloader(opt)
    # 模型
    model = get_model(opt)
    # 参数
    loss_fn = get_loss(opt)
    optimizer = torch.optim.Adam(model.parameters())  # 优化器
    # 训练
    for epoch in range(opt.epochs):
        opt.epoch = epoch + 1  # 设置当前循环轮次
        train(train_dataloader, model, loss_fn, optimizer, opt)  # 训练
        test(test_dataloader, model, loss_fn, opt)  # 测试
    # 保存
    torch.save(model.state_dict(), opt.save_path)
    print(f'Saved PyTorch Model State to {opt.save_path}')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
