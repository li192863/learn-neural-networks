import argparse
import time

import torch

import utils
from dataset import FaceMaskDataset
from model import get_model_object_detection
from engine import train_one_epoch, evaluate
from presets import DetectionPresetTrain, DetectionPresetEval

DATASET_ROOT_PATH = '../../../datasets/face-mask-detection'
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 1
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'data/model.pth'
DEFAULT_WORKERS = 16
classes = ['background', 'with_mask', 'without_mask', 'mask_weared_incorrect']


def get_dataloader(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    # 使用数据集
    train_data = FaceMaskDataset(DATASET_ROOT_PATH, DetectionPresetTrain(data_augmentation='hflip'))
    test_data = FaceMaskDataset(DATASET_ROOT_PATH, DetectionPresetEval())

    # 划分数据集为训练集与测试集
    indices = torch.randperm(len(train_data)).tolist()
    train_data = torch.utils.data.Subset(train_data, indices[:-50])
    test_data = torch.utils.data.Subset(test_data, indices[-50:])

    # 定义数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                                   collate_fn=utils.collate_fn)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                                  num_workers=opt.workers, collate_fn=utils.collate_fn)
    return train_dataloader, test_dataloader


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
    model = get_model_object_detection(num_classes).to(opt.device)
    # 参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)  # 优化器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decays the learning rate of each parameter group by gamma every step_size epochs.
    # 训练
    for epoch in range(opt.epochs):
        opt.epoch = epoch  # 设置当前循环轮次
        train_one_epoch(model, optimizer, train_dataloader, opt.device, opt.epoch, print_freq=10)  # 训练
        lr_scheduler.step()  # 更新学习率
        evaluate(model, test_dataloader, opt.device)  # 测试
    print(f'Done!')
    # 保存
    torch.save(model.state_dict(), opt.save_path)
    print(f'Saved PyTorch Model State to {opt.save_path}')
    # 计时
    show_time_elapse(start, time.time(), 'Training complete in')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

    # # 测试模型
    # model = get_model_object_detection(4)
    # dataset = FaceMaskDataset('../../datasets/face-mask-detection', get_transform(train=True))
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4,
    #     collate_fn=utils.collate_fn
    # )
    # # 测试训练
    # images, targets = next(iter(data_loader))
    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    # output = model(images, targets)  # Returns losses and detections
    # print(output)
    # # 测试推理
    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)  # Returns predictions
    # print(predictions)
