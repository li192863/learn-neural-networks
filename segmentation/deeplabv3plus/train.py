import argparse
import time
from functools import partial

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from dataset import WSISegmentationDataset
# from engine import train_one_epoch, evaluate, criterion
from presets import SegmentationPresetTrain, SegmentationPresetEval
from model.deeplabv3plus import DeepLabV3Plus

DATASET_ROOT_PATH = r'E:\Projects\Carcinoma\Data\Segmentation\分割数据集-最终'
DEFAULT_EPOCHS = 2  # 200
DEFAULT_BATCH_SIZE = 2
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'data/model.pth'
DEFAULT_WORKERS = 8
classes = ['_background_', 'Normal', 'Tumor']


def get_dataloader(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    # 使用数据集
    train_data = WSISegmentationDataset(DATASET_ROOT_PATH, type='train', transforms=SegmentationPresetTrain(base_size=512, crop_size=512))
    test_data = WSISegmentationDataset(DATASET_ROOT_PATH, type='val', transforms=SegmentationPresetEval(base_size=512))

    # # 划分数据集为训练集与测试集
    # torch.manual_seed(19990924)
    # indices = torch.randperm(len(train_data)).tolist()
    # train_data = torch.utils.data.Subset(train_data, indices[:-10])
    # test_data = torch.utils.data.Subset(test_data, indices[-10:])

    # 定义数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=opt.workers)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                                  num_workers=opt.workers)
    return train_dataloader, test_dataloader


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
            pbar.set_postfix({'loss': f'{total_loss:>7f}', 'lr': f'{optimizer.param_groups[0]["lr"]}'})


def test(dataloader, model, loss_fn, opt, num_classes):
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
    loss, confmat = 0, utils.ConfusionMatrix(num_classes)
    model.eval()  # Sets the module in evaluation mode
    with torch.no_grad():  # Disabling gradient calculation
        with tqdm(dataloader, desc=' ' * (len(str(opt.epoch)) + len(str(opt.epochs)) + 9) + 'test',
                  total=len(dataloader)) as pbar:  # 进度条
            for X, y in pbar:
                X, y = X.to(opt.device), y.to(opt.device)  # 载入数据
                pred = model(X)  # 预测结果
                loss += loss_fn(pred, y).item()  # 计算损失
                confmat.update(y.flatten(), pred.argmax(1).flatten())  # 更新指标
                pbar.set_postfix({'Avg loss': f'{loss / num_batches:>8f}'})
    return loss, confmat  # 返回准确率


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
    model = DeepLabV3Plus(num_classes, pretrained=True).to(opt.device)
    # model.load_state_dict(torch.load(opt.save_path)) if os.path.exists(opt.save_path) else ''
    # 参数
    # params = [
    #     {'params': [p for p in model.backbone.parameters() if p.requires_grad]},
    #     {'params': [p for p in model.classifier.parameters() if p.requires_grad]}
    # ]
    loss_fn = partial(nn.functional.cross_entropy, ignore_index=255)  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)  # 优化器
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=len(train_dataloader) * opt.epochs,
                                                         power=0.9)  # Decays the learning rate of each parameter group using a polynomial function in the given total_iters.
    # 训练
    writer = SummaryWriter('data/tensorboard')
    best_acc = 0.0
    for epoch in range(opt.epochs):
        opt.epoch = epoch  # 设置当前循环轮次
        train(train_dataloader, model, loss_fn, optimizer, opt)  # 训练
        lr_scheduler.step()  # 更新学习率
        loss, confmat = test(test_dataloader, model, loss_fn, opt, num_classes)  # 测试
        acc = confmat.compute()[0].item()
        if acc > best_acc:
            best_acc = max(acc, best_acc)
            # 保存
            torch.save(model.state_dict(), opt.save_path)
            print(f'Saved PyTorch Model State to {opt.save_path}, model\'s best accuracy is {100 * best_acc:>0.1f}%')
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('accuracy', acc, epoch)
        writer.add_scalar('learning rate', lr_scheduler.get_last_lr()[0], epoch)
    writer.close()
    print(f'Done!')
    # 保存
    torch.save(model.state_dict(), 'data/model_last.pth')
    print(f'Saved PyTorch Model State to data/model_last.pth, model\'s best accuracy is {100 * best_acc:>0.1f}%')
    # 计时
    show_time_elapse(start, time.time(), 'Training complete in')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

    # # 测试模型
    # model = get_model_sematic_segmentation(6)
    # dataset = FootballPlayerSegmentationDataset(DATASET_ROOT_PATH, transforms=SegmentationPresetTrain(base_size=520, crop_size=480))
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4,
    #     collate_fn=utils.collate_fn
    # )
    # # 测试训练
    # images, targets = next(iter(data_loader))
    # output = model(images)  # Returns losses and detections
    # loss = criterion(output, targets)
    # print(loss)
    # # 测试推理
    # model.eval()
    # x = torch.rand(2, 3, 300, 400)
    # predictions = model(x)  # Returns predictions
    # print(predictions)
