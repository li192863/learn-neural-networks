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
DEFAULT_BATCH_SIZE = 10
DEFAULT_BPTT = 35  # sequence length
DEFAULT_PREDICT_SIZE = 10
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL_PATH = 'data/model.pth'
DEFAULT_WORKERS = 8
emsize = 200  # size of word embeddings
nhead = 2  # the number of heads in the encoder/decoder of the transformer model
nhid = 200  # number of hidden units per layer
nlayers = 2  # number of layers
dropout = 0.2  # dropout applied to layers (0 = no dropout)


def get_test_data(opt):
    """
    获取数据加载器
    :param opt:
    :return:
    """
    corpus = Corpus(DATASET_ROOT_PATH)
    test_data = batchify(corpus.test, opt.batch_size).to(opt.device)

    x, y = get_batch(test_data, torch.randint(0, len(range(0, test_data.size(0) - 1, opt.bptt)), ()).item(), bptt=opt.bptt)
    return x, y, corpus


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


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model data path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--bptt', type=int, default=DEFAULT_BPTT, help='sequence length')
    parser.add_argument('--predict-size', type=int, default=DEFAULT_PREDICT_SIZE, help='predicted sequence length')
    parser.add_argument('--device', default=DEFAULT_DEVICE, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', default=DEFAULT_WORKERS, help='max dataloader workers')
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
    x, y, corpus = get_test_data(opt)
    # 模型
    ntokens = len(corpus.dictionary)
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(opt.device)
    # 参数
    model.load_state_dict(torch.load(opt.model_path))
    # 评估
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        output = model(x)

        sequences = [' '.join([corpus.dictionary.idx2word[j] for j in x[:, i]]) for i in range(opt.batch_size)]
        predict_idx = torch.argmax(torch.softmax(output.view(-1, ntokens), dim=-1), dim=-1)[-opt.batch_size:]
        predict = [corpus.dictionary.idx2word[i] for i in predict_idx]
        actual = [corpus.dictionary.idx2word[i] for i in y[-opt.batch_size:]]

        for i, seq in enumerate(sequences):
            print(seq, '[', predict[i], ']', '(' + actual[i] + ')' if actual[i] != predict[i] else '')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
