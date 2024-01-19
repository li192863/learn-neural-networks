import argparse
import os
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet18
from tqdm import tqdm

from presets import ClassificationPresetTrain, ClassificationPresetEval

DATASET_ROOT_PATH = '../../../datasets/hymenoptera_data'
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'data/model.pth'
DEFAULT_WORKERS = 16
classes = ['ants', 'bees']


def get_dataloader(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    training_data = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_PATH, 'train'),
                                         transform=ClassificationPresetTrain(crop_size=224))
    test_data = datasets.ImageFolder(root=os.path.join(DATASET_ROOT_PATH, 'val'),
                                     transform=ClassificationPresetEval(crop_size=224))
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.workers)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=opt.batch_size, num_workers=opt.workers)
    return train_dataloader, test_dataloader


def get_model_finetuning_the_convnet(num_classes):
    """
    获取卷积层微调的resnet18模型
    :param num_classes:
    :return:
    """
    model = resnet18(weights='DEFAULT')  # 下载预训练权重

    # 微调卷积层
    in_features = model.fc.in_features  # 获取全连接层输入种类数
    model.fc = nn.Linear(in_features, num_classes)  # 更改全连接层

    return model


def get_model_convnet_as_fixed_feature_extractor(num_classes):
    """
    获取卷积层固定的resnet18模型
    :param num_classes:
    :return:
    """
    model = resnet18(weights='DEFAULT')  # 下载预训练权重

    # 冻结卷积层
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features  # 获取全连接层输入种类数
    model.fc = nn.Linear(in_features, num_classes)  # 更改全连接层（注意仅最后层的全连接层权重被优化）

    return model


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
    total_loss = 0.0
    model.train()  # Sets the module in training mode
    with tqdm(dataloader, desc=f'Epoch {opt.epoch}/{opt.epochs}, train', total=len(dataloader)) as pbar:  # 进度条
        for X, y in pbar:
            # 前向传播
            X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
            pred = model(X)  # 预测结果
            loss = loss_fn(pred, y)  # 计算损失

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            # 打印信息
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{total_loss:>7f}'})


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
    loss, correct = 0, 0
    model.eval()  # Sets the module in evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        with tqdm(dataloader, desc=' ' * (len(str(opt.epoch)) + len(str(opt.epochs)) + 9) + 'test',
                  total=len(dataloader)) as pbar:  # 进度条
            for X, y in pbar:
                X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
                pred = model(X)  # 预测结果
                loss += loss_fn(pred, y).item()  # 计算损失
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 判断正误
                pbar.set_postfix(
                    {'Accuracy': f'{(100 * correct / size):>0.1f}%', 'Avg loss': f'{loss / num_batches:>8f}'})
    return correct / size  # 返回准确率


def show_time_elapse(start, end=None, prefix='', suffix=''):
    """
    显示运行时间
    :param start:
    :param end:
    :param prefix:
    :param suffix:
    :return:
    """
    end = end or time.time()
    time_elapsed = end - start  # 单位为秒
    hours = time_elapsed // 3600  # 时
    minutes = (time_elapsed - hours * 3600) // 60  # 分
    seconds = (time_elapsed - hours * 3600 - minutes * 60) // 1  # 秒
    if hours == 0:  # 0 hours x minutes x seconds
        if minutes == 0:  # 0 hours 0 minutes x seconds
            print(prefix + f' {seconds:.0f}s ' + suffix)
        else:  # 0 hours x minutes x seconds
            print(prefix + f' {minutes:.0f}m {seconds:.0f}s ' + suffix)
    else:  # x hours x minutes x seconds
        print(prefix + f' {hours:.0f}h {minutes:.0f}m {seconds:.0f}s ' + suffix)


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
    parser.add_argument('--workers', default=DEFAULT_WORKERS, help='max dataloader workers')
    return parser.parse_args()


def main(opt):
    """
    主函数
    :param opt:
    :return:
    """
    # 计时
    start = time.time()
    # 设备
    print(f'Using {opt.device} device')
    # 数据
    train_dataloader, test_dataloader = get_dataloader(opt)
    # 模型
    num_classes = len(classes)
    # model = NeuralNetwork(num_classes).to(opt.device)  # 权值随机值模型 best_acc = 72.5%
    # model = get_model_finetuning_the_convnet(num_classes).to(opt.device)  # 微调卷积层模型 best_acc = 96.1%
    model = get_model_convnet_as_fixed_feature_extractor(num_classes).to(opt.device)  # 冻结卷积层模型 best_acc = 96.7%
    # 参数
    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 优化器
    lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    # 训练
    best_acc = 0.0
    for epoch in range(opt.epochs):
        opt.epoch = epoch + 1  # 设置当前循环轮次
        train(train_dataloader, model, loss_fn, optimizer, opt)  # 训练
        lr_scheduler.step()  # 更新学习率
        acc = test(test_dataloader, model, loss_fn, opt)  # 测试

        best_acc = max(acc, best_acc)  # 获取准确率
    # 保存
    torch.save(model.state_dict(), opt.save_path)
    print(f'Saved PyTorch Model State to {opt.save_path}, model\'s best accuracy is {100 * best_acc:>0.1f}%')
    # 计时
    show_time_elapse(start, time.time(), 'Training complete in')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
