import argparse
import time
import torch
import torch.nn as nn
import torch.onnx
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from word_language.dataset import Corpus

from model import TransformerModel

DATASET_ROOT_PATH = '../../datasets/wikitext-2'
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 20
DEFAULT_EVAL_BATCH_SIZE = 10
DEFAULT_BPTT = 35  # sequence length
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAVE_PATH = 'data/model.pth'
DEFAULT_WORKERS = 8
emsize = 200  # size of word embeddings
nhead = 2  # the number of heads in the encoder/decoder of the transformer model
nhid = 200  # number of hidden units per layer
nlayers = 2  # number of layers
dropout = 0.2  # dropout applied to layers (0 = no dropout)


def get_data(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    corpus = Corpus(DATASET_ROOT_PATH)
    train_data = batchify(corpus.train, opt.batch_size).to(opt.device)
    val_data = batchify(corpus.valid, opt.eval_batch_size).to(opt.device)
    test_data = batchify(corpus.test, opt.eval_batch_size).to(opt.device)
    return corpus, train_data, val_data, test_data


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    """
    将data按bsz生成batch
    @param data: 原始数据
    @param bsz: 一个batch的大小
    @return: (num_batches, batch_size)
    """
    # 将数据按bsz大小划分 Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # 移除数据集多余部分（残留部分） Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # 将数据集按bsz大小生成batch Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, bptt=35):
    """
    获取一个批次
    @param source: 源数据，通常是一个序列（如文本）的编码表示
    @param i: 当前批次的起始位置
    @param bptt: 用于截断反向传播（Backpropagation Through Time）的长度
    @return: 一个批次的输入数据（从源数据中截取的一部分）, 与data对应的目标数据（即data的下一个元素）
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def train(corpus, data_source, model, loss_fn, optimizer, opt):
    """
    训练模型
    :param corpus:
    :param data_source:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param opt:
    :return:
    """
    ntokens = len(corpus.dictionary)
    model.train()  # Turn on training mode which enables dropout.
    with tqdm(range(0, data_source.size(0) - 1, opt.bptt), desc=f'Epoch {opt.epoch}/{opt.epochs}, train',
              total=len(range(0, data_source.size(0) - 1, opt.bptt))) as pbar:  # 进度条
        for batch, i in enumerate(pbar):
            # 前向传播
            data, targets = get_batch(data_source, i, bptt=opt.bptt)  # 载入数据
            output = model(data).view(-1, ntokens)  # 预测结果
            loss = loss_fn(output, targets)  # 计算损失
            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 防止梯度爆炸

            # 打印信息
            pbar.set_postfix({'loss': f'{loss.item():>7f}'})


def test(corpus, data_source, model, loss_fn, opt):
    """
    测试模型
    :param corpus:
    :param data_source:
    :param model:
    :param loss_fn:
    :param opt:
    :return:
    """
    num_batches = len(range(0, data_source.size(0) - 1, opt.bptt))
    loss = 0.
    ntokens = len(corpus.dictionary)
    model.eval()  # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        with tqdm(range(0, data_source.size(0) - 1, opt.bptt), desc=' ' * (len(str(opt.epoch)) + len(str(opt.epochs)) + 9) + 'test',
                  total=len(range(0, data_source.size(0) - 1, opt.bptt))) as pbar:  # 进度条
            for batch, i in enumerate(pbar):
                data, targets = get_batch(data_source, i, bptt=opt.bptt)
                output = model(data).view(-1, ntokens)
                loss += len(data) * loss_fn(output, targets).item()
                pbar.set_postfix({'Avg loss': f'{loss / num_batches:>8f}'})
    return loss / (len(data_source) - 1)


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
    parser.add_argument('--eval-batch-size', type=int, default=DEFAULT_EVAL_BATCH_SIZE, help='eval batch size')
    parser.add_argument('--bptt', type=int, default=DEFAULT_BPTT, help='sequence length')
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
    corpus, train_data, val_data, test_data = get_data(opt)
    # 模型
    ntokens = len(corpus.dictionary)
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(opt.device)
    # 参数
    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 优化器
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    # 训练
    best_val_loss = float('inf')
    for epoch in range(opt.epochs):
        opt.epoch = epoch + 1  # 设置当前循环轮次
        train(corpus, train_data, model, loss_fn, optimizer, opt)  # 训练
        lr_scheduler.step()  # 更新学习率
        val_loss = test(corpus, val_data, model, loss_fn, opt)  # 测试

        best_val_loss = min(val_loss, best_val_loss)  # 获取损失
    # 保存
    torch.save(model.state_dict(), opt.save_path)
    print(f'Saved PyTorch Model State to {opt.save_path}, model\'s best_val_loss is {best_val_loss:>0.2f}')
    # 计时
    show_time_elapse(start, time.time(), 'Training complete in')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
