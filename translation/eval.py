import argparse
import copy
import torch
import torch.onnx
from torch.utils.data import random_split, DataLoader

from translation.dataset import TranslationDataset, en_tokenizer, zh_tokenizer, Dictionary, Corpus
from model import TranslationModel

DATASET_ROOT_PATH = '../../datasets/en-to-zh'
DEFAULT_BATCH_SIZE = 10
DEFAULT_PAD_IDX = 2
DEFAULT_PREDICT_SIZE = 10
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_MODEL_PATH = 'data/model.pth'
DEFAULT_WORKERS = 8


def get_test_data(opt):
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

    train_data, test_data = random_split(dataset, [train_size, test_size], torch.Generator().manual_seed(777))
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True)

    x, y = next(iter(copy.deepcopy(test_dataloader)))
    x, y = x.to(opt.device), y.to(opt.device)
    return x, y, dataset


def parse_opt():
    """
    解析命令行参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='model data path')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='batch size')
    parser.add_argument('--pad-idx', type=int, default=DEFAULT_PAD_IDX, help='pad index of the sequences')
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
    x, y, dataset = get_test_data(opt)
    # 模型
    d_model = 512
    model = TranslationModel(d_model, dataset.src_dictionary, dataset.tgt_dictionary, pad_idx=opt.pad_idx, max_len=30).to(
        opt.device)
    # 参数
    model.load_state_dict(torch.load(opt.model_path))
    # 评估
    model.eval()  # Sets the module in training mode.
    with torch.no_grad():  # Disabling gradient calculation
        y_x, y_y = y[:, :-1], y[:, 1:]
        output = model(x, y_x)

        # 获取特殊标志的序号
        bos_idx = dataset.tgt_dictionary.word2idx['<s>']  # 开始
        eos_idx = dataset.tgt_dictionary.word2idx['</s>']  # 结束
        pad_idx = dataset.tgt_dictionary.word2idx['<pad>']  # 填充
        useless_tokens = [bos_idx, eos_idx, pad_idx]

        sequences = [' '.join([dataset.src_dictionary.idx2word[j] if j not in useless_tokens else ''
                               for j in x[i, :]]).strip().replace(' ##', '') for i in range(opt.batch_size)]

        predict_idx = torch.argmax(output, dim=-1)
        predict = []
        for i in range(opt.batch_size):
            line = ''
            for j in predict_idx[i, :]:
                if j == eos_idx: break
                line += dataset.tgt_dictionary.idx2word[j]
            predict.append(line)

        actual = [''.join([dataset.tgt_dictionary.idx2word[j] if j not in useless_tokens else ''
                            for j in y_y[i, :]]).strip() for i in range(opt.batch_size)]

        for i, seq in enumerate(sequences):
            print('Origin:', seq)
            print('Actual: ', actual[i])
            print('Predict: ', predict[i])


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
