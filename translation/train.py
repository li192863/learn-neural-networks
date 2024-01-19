import argparse
import time
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import TranslationModel, TranslationLoss, CustomSchedule
from translation.dataset import TranslationDataset, en_tokenizer, zh_tokenizer, collate_fn, Dictionary, Corpus

DATASET_ROOT_PATH = '../../datasets/en-to-zh'
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_PAD_IDX = 2
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'data/model.pth'
DEFAULT_WORKERS = 8


def get_dataloader(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    train_ratio = 0.99  # 训练集占比
    test_ratio = 1 - train_ratio  # 测试集占比

    dataset = TranslationDataset(DATASET_ROOT_PATH, 'en', 'zh', en_tokenizer, zh_tokenizer, max_len=30)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)  # 训练集大小
    test_size = total_size - train_size  # 测试集大小

    torch.manual_seed(0)
    train_data, test_data = random_split(dataset, [train_size, test_size], torch.Generator().manual_seed(777))

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.workers)
    return train_dataloader, test_dataloader, dataset


def train(dataloader, model, loss_fn, optimizer, lr_scheduler, opt):
    """
    训练模型
    :param dataloader:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param lr_scheduler: 基于step变化的lr_scheduler
    :param opt:
    :return:
    """
    model.train()  # Turn on training mode which enables dropout.
    with tqdm(dataloader, desc=f'Epoch {opt.epoch}/{opt.epochs}, train', total=len(dataloader)) as pbar:  # 进度条
        for i, (src, tgt) in enumerate(pbar):
            tgt_x, tgt_y = tgt[:, :-1], tgt[:, 1:]
            # 前向传播
            src, tgt_x, tgt_y = src.to(opt.device), tgt_x.to(opt.device), tgt_y.to(opt.device)  # 载入数据
            out = model(src, tgt_x)  # 预测结果 # (b, tgt_seq_len, len(tgt_vocab))
            loss = loss_fn(out, tgt_y)  # 计算损失

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            # 更新学习率-基于step
            lr_scheduler.step()  # 更新学习率

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
    num_batches = len(dataloader)
    loss, correct = 0., 0.
    model.eval()  # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        with tqdm(dataloader, desc=' ' * (len(str(opt.epoch)) + len(str(opt.epochs)) + 9) + 'test',
                  total=len(dataloader)) as pbar:  # 进度条
            for i, (src, tgt) in enumerate(pbar):
                tgt_x, tgt_y = tgt[:, :-1], tgt[:, 1:]
                src, tgt_x, tgt_y = src.to(opt.device), tgt_x.to(opt.device), tgt_y.to(opt.device)  # 载入数据
                pred = model(src, tgt_x)  # 预测结果 (b, tgt_seq_len, len(tgt_vocab))
                loss += loss_fn(pred, tgt_y).item()  # 计算损失
                correct += get_acc(pred, tgt_y, pad_idx=opt.pad_idx).item()  # 判断正误
                pbar.set_postfix(
                    {'Accuracy': f'{(100 * correct / num_batches):>0.1f}%', 'Avg loss': f'{loss / num_batches:>8f}'})
    return correct / num_batches  # 返回准确率


def get_acc(pred, target, pad_idx=2):
    """
    计算ACC
    @param pred:
    @param target:
    @param pad_idx:
    @return:
    """
    corrects = torch.eq(pred.argmax(dim=-1), target)
    # 处理mask
    mask = torch.logical_not(target.eq(pad_idx))
    corrects *= mask
    return corrects.sum() / mask.sum()


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
    parser.add_argument('--pad-idx', type=int, default=DEFAULT_PAD_IDX, help='pad index of the sequences')
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
    train_dataloader, test_dataloader, dataset = get_dataloader(opt)
    # 模型
    d_model = 512
    model = TranslationModel(d_model, dataset.src_dictionary, dataset.tgt_dictionary, pad_idx=opt.pad_idx, max_len=30).to(
        opt.device)
    # 参数
    loss_fn = TranslationLoss(pad_idx=opt.pad_idx)  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)  # 优化器
    lr_scheduler = CustomSchedule(optimizer, d_model, warm_steps=4000)  # lr_scheduler基于step变化
    # 训练
    best_acc = 0.0
    for epoch in range(opt.epochs):
        opt.epoch = epoch + 1  # 设置当前循环轮次
        train(train_dataloader, model, loss_fn, optimizer, lr_scheduler, opt)
        # lr_scheduler.step()  # 更新学习率
        acc = test(test_dataloader, model, loss_fn, opt)

        if acc > best_acc: torch.save(model.state_dict(), 'data/model_best.pth')
        best_acc = max(acc, best_acc)  # 获取准确率
    # 保存
    torch.save(model.state_dict(), opt.save_path)
    print(f'Saved PyTorch Model State to {opt.save_path}, model\'s best_acc is {best_acc:>0.2f}')
    # 计时
    show_time_elapse(start, time.time(), 'Training complete in')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
