import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from translation.dataset import Corpus, en_tokenizer, zh_tokenizer, TranslationDataset, Dictionary, Corpus


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置 [[0], [1], [2], ..., [max_len - 1]], (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 角度 (ceil(d_model / 2),)
        pe[:, 0::2] = torch.sin(position * div_term)  # 2i: 0, 2, 4, 6, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 2i+1: 1, 3, 5, 7, ...
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 注册名为'pe'的缓冲区，存放位置编码

    def forward(self, x):  # (seq_len, N, d_model)
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        assert self.pe.size(1) >= x.size(1)
        x = x + self.pe[:, :x.size(1)] # (N, seq_len, d_model) + (1, sql_len, d_model)
        return self.dropout(x)  # (N, seq_len, d_model)


class TranslationModel(nn.Module):
    """ 翻译模型 """
    pad_idx = 0
    def __init__(self, d_model, src_vocab, tgt_vocab, pad_idx=2, dropout=0.1, max_len=50):
        super(TranslationModel, self).__init__()
        TranslationModel.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)

        self.final_layer = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):  # (b, src_seq_len)  (b, tgt_seq_len)
        # 获取蒙版
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(src.device)  # look ahead mask (tgt_seq_len, tgt_seq_len)
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)  # src padding mask (b, src_seq_len)
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)  # tgt padding mask (b, tgt_seq_len)

        src = self.src_embedding(src)  # (b, src_seq_len, d_model)
        src = self.pos_encoder(src)  # (b, src_seq_len, d_model)

        tgt = self.tgt_embedding(tgt)  # (b, tgt_seq_len, d_model)
        tgt = self.pos_encoder(tgt)  # (b, tgt_seq_len, d_model)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # (b, tgt_seq_len, d_model)

        out = self.final_layer(out)

        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        生成true/false的mask
        @param tokens: int tensor
        @return: bool tensor
        """
        return tokens == TranslationModel.pad_idx


class TranslationLoss(nn.Module):
    """ 翻译损失 """
    def __init__(self, pad_idx=2):
        super(TranslationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.pad_idx = pad_idx

    def forward(self, pred, target):  # (b, tgt_seq_len, len(tgt_vocab)), (b, tgt_seq_len)
        """ 计算损失 """
        loss = self.criterion(pred.transpose(-1, -2), target)  # (b, tgt_seq_len)
        # 处理mask，若不考虑mask则损失会偏大
        mask = torch.logical_not(target.eq(self.pad_idx)).type(torch.float32)  # (b, targ_seq_len)
        loss *= mask  # 非pad处值不变，pad处值置0 (b, tgt_seq_len)
        return loss.sum() / mask.sum()


class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    """ 自定义学习率 """
    def __init__(self, optimizer, d_model, warm_steps=4):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warm_steps
        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        """ 学习率 """
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        dynamic_lr = (self.d_model ** (-0.5)) * min(arg1, arg2)
        return [dynamic_lr for group in self.optimizer.param_groups]


if __name__ == '__main__':
    dataset = TranslationDataset('../../datasets/en-to-zh', 'en', 'zh', src_tokenizer=en_tokenizer, tgt_tokenizer=zh_tokenizer)
    model = TranslationModel(256, dataset.src_dictionary, dataset.tgt_dictionary)
    print(model)
    src = torch.randint(0, 79, (20, 30))  # 单词数 >> 序列长
    tgt = torch.randint(0, 83, (20, 40))  # 单词数 >> 序列长
    tgt_y = torch.randint(0, 83, (20, 40))  # 单词数 >> 序列长
    output = model(src, tgt)  # (b, tgt_seq_len, d_model)
    print(output.shape)

    loss_fn = TranslationLoss()
    loss = loss_fn(torch.load('data/fake_out.pt'), torch.load('data/fake_tgt_y.pt'))
    print(loss)