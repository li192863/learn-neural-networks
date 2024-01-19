import os
from io import open
import torch


class Dictionary(object):
    """ 字典 """
    def __init__(self):
        self.word2idx = {}  # 单词 -> 下标
        self.idx2word = []  # 下标 -> 单词

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
    def __init__(self, path):
        self.dictionary = Dictionary()  # 字典
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """ 将一个文本文件代码化 Tokenizes a text file."""
        assert os.path.exists(path)
        # 往字典里添加单词 Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # 文件内容代码化 Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids

if __name__ == '__main__':
    c = Corpus('../../datasets/wikitext-2')
    print(len(c.dictionary))
    print(c.train.shape)
    print(c.test.shape)
    print(c.valid.shape)
