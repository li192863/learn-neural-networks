import os
from io import open

import jieba
import torch
from torch.nn.functional import pad
from tqdm import tqdm
from transformers import BertTokenizer


class Dictionary(object):
    """ 字典 """
    def __init__(self):
        self.word2idx = {}  # 单词 -> 下标
        self.idx2word = []  # 下标 -> 单词

        # 特殊词汇
        self.specials = ["<s>", "</s>", "<pad>", "<unk>"]
        for special in self.specials:
            self.add_word(special)
        self.bos_idx = self.word2idx['<s>']  # 开始 0
        self.eos_idx = self.word2idx['</s>']  # 结束 1
        self.pad_idx = self.word2idx['<pad>']  # 填充 2
        self.unk_idx = self.word2idx['<unk>']  # 低频 3


    def add_word(self, word):
        """ 添加单词 """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]  # 返回单词下标

    def __len__(self):
        """ 字典长度 """
        return len(self.idx2word)


class Corpus(object):
    """ 语料库 """
    def __init__(self, path, tokenizer, max_len=30):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dictionary = Dictionary()  # 字典
        self.list = self.tokenize(path)

    def tokenize(self, path):
        """ 将一个文本文件代码化 Tokenizes a text file."""
        assert os.path.exists(path)
        file = open(path, 'r', encoding='utf8')
        lines = file.readlines()
        file.close()
        num_lines = len(lines)
        # 往字典里添加单词 Add words to the dictionary ~400s
        with tqdm(lines, desc=f'update dictionary from file {path}', total=num_lines) as pbar:
            for line in pbar:
                words = self.tokenizer(line)
                for word in words:
                    self.dictionary.add_word(word)

        # 获取特殊标志的序号
        bos_idx = torch.tensor([self.dictionary.word2idx['<s>']])  # 开始
        eos_idx = torch.tensor([self.dictionary.word2idx['</s>']])  # 结束
        pad_idx = self.dictionary.word2idx['<pad>']  # 填充

        # 文件内容代码化 Tokenize file content ~400s
        idss = []
        with tqdm(lines, desc=f'update corpus from file {path}', total=num_lines) as pbar:
            for line in pbar:
                words = self.tokenizer(line)
                origin = [self.dictionary.word2idx[word] for word in words]
                add_bos_eos = torch.cat([bos_idx, torch.tensor(origin).type(torch.int64), eos_idx], 0)
                processed = pad(add_bos_eos, (0, self.max_len - len(add_bos_eos)), value=pad_idx)
                if processed[-1] != pad_idx: processed[-1] = eos_idx  # 保证每句话都有bos和eos
                idss.append(processed)
        ids = torch.stack(idss)  # (len(list), max_length) ~45s
        return ids


class TranslationDataset(torch.utils.data.Dataset):
    """ 翻译数据集 """
    def __init__(self, root, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, max_len=30, use_cache=True):
        self.root = root
        self.use_cache = use_cache
        self.max_len = max_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_raw_text = os.path.join(self.root, f'train.{src_lang}')
        self.tgt_raw_text = os.path.join(self.root, f'train.{tgt_lang}')
        self.src_dictionary_cache_file = os.path.join(self.root, f'train.{src_lang}.dictionary.pt')
        self.tgt_dictionary_cache_file = os.path.join(self.root, f'train.{tgt_lang}.dictionary.pt')
        self.src_corpus_cache_file = os.path.join(self.root, f'train.{src_lang}.corpus.pt')
        self.tgt_corpus_cache_file = os.path.join(self.root, f'train.{tgt_lang}.corpus.pt')

        self.src_dictionary, self.tgt_dictionary, self.src_corpus, self.tgt_corpus = None, None, None, None
        self.load_data()  # 加载语料库和字典

    def load_data(self):
        """ 加载字典与语料库 """
        if self.use_cache:
            if os.path.exists(self.src_dictionary_cache_file):
                self.src_dictionary = torch.load(self.src_dictionary_cache_file)
                print(f'Load source dictionary from cache file \'{self.src_dictionary_cache_file}\'.')
            if os.path.exists(self.tgt_dictionary_cache_file):
                self.tgt_dictionary = torch.load(self.tgt_dictionary_cache_file)
                print(f'Load target dictionary from cache file \'{self.tgt_dictionary_cache_file}\'.')
            if os.path.exists(self.src_corpus_cache_file):
                self.src_corpus = torch.load(self.src_corpus_cache_file)
                print(f'Load source corpus from cache file \'{self.src_corpus_cache_file}\'.')
            if os.path.exists(self.tgt_corpus_cache_file):
                self.tgt_corpus = torch.load(self.tgt_corpus_cache_file)
                print(f'Load target corpus from cache file \'{self.tgt_corpus_cache_file}\'.')
        if self.src_dictionary is None or self.src_corpus is None:
            src_corpus = Corpus(self.src_raw_text, self.src_tokenizer, max_len=self.max_len)  # ~450s
            self.src_dictionary = src_corpus.dictionary
            self.src_corpus = src_corpus.list
            torch.save(self.src_dictionary, self.src_dictionary_cache_file)
            torch.save(self.src_corpus, self.src_corpus_cache_file)
        if self.tgt_dictionary is None or self.tgt_corpus is None:
            tgt_corpus = Corpus(self.tgt_raw_text, self.tgt_tokenizer, max_len=self.max_len)
            self.tgt_dictionary = tgt_corpus.dictionary
            self.tgt_corpus = tgt_corpus.list
            torch.save(self.tgt_dictionary, self.tgt_dictionary_cache_file)
            torch.save(self.tgt_corpus, self.tgt_corpus_cache_file)

    def __getitem__(self, item):
        return self.src_corpus[item], self.tgt_corpus[item]

    def __len__(self):
        return self.src_corpus.size(0)


def en_tokenizer(line):
    """ 英文分词器 """
    # 加载基础的分词器模型，使用的是基础的bert模型。`uncased`意思是不区分大小写
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符
    return tokenizer.tokenize(line)


def zh_tokenizer(line):
    """ 中文分词器 """
    return list(line.strip().replace(' ', ''))


def collate_fn(batch):
    """ 收集器 """
    src = torch.stack([item[0] for item in batch])
    tgt = torch.stack([item[1] for item in batch])
    tgt_y = tgt[:, 1:]  # tgt_y是目标句子去掉第一个token，即去掉<bos>
    tgt = tgt[:, :-1]  # tgt是目标句子去掉最后一个token
    return src, tgt, tgt_y


if __name__ == '__main__':
    print(en_tokenizer('A pair of nines? You pushed in with a pair of nines?'))
    print(zh_tokenizer('你好哈？'))
    root = '../../datasets/en-to-zh'
    t = TranslationDataset(root, 'en', 'zh', src_tokenizer=en_tokenizer, tgt_tokenizer=zh_tokenizer)
    print(' '.join([t.src_dictionary.idx2word[i] for i in t[7][0]]))
    print(' '.join([t.tgt_dictionary.idx2word[i] for i in t[7][1]]))
    print(t[4])
    print('wait!')
