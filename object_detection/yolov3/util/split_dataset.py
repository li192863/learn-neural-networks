import os
import random
import argparse


def parse_args():
    """
    传入命令行参数
    :return: 参数
    """
    parser = argparse.ArgumentParser()
    # xml文件(原始注解)的地址(annotations)
    parser.add_argument('--xml_path', default='annotations',
                        type=str, help='input xml label path')
    # txt文件(划分后路径名)的地址(dataset_path)
    parser.add_argument('--dataset_path', default='datasets',
                        type=str, help='output path of splited file paths')
    opt = parser.parse_args()
    return opt


def generate_index(xml_path, train=0.8, val=0.1):
    """
    生成训练集/验证集/测试集索引
    :param xml_path: 原始注解路径地址
    :param train: 训练集占数据集划分比
    :param val: 验证集占数据集划分比
    :return: xml文件列表 训练集与验证集索引列表 训练集索引列表
    """
    total_xml = os.listdir(xml_path)
    num = len(total_xml)
    list_index = range(num)
    tv = int(num * (train + val))
    tr = int(tv * val / (train + val))

    trainval_idx = random.sample(list_index, tv)  # 训练集 + 验证集
    val_idx = random.sample(trainval_idx, tr)  # 验证集
    return total_xml, trainval_idx, val_idx


def generate_txt(dataset_path, total_xml, trainval_idx, val_idx, prefix='', suffix=''):
    """
    生成txt文件
    :param dataset_path: 划分后存放txt文件的路径地址
    :param total_xml: xml文件列表
    :param trainval_idx: 训练集与验证集索引列表
    :param val_idx: 验证集索引列表
    :param prefix: txt文件中图片名称前缀
    :param suffix: txt文件中图片名称后缀
    """
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # file_trainval = open(txt_path + '/trainval.txt', 'w')
    file_test = open(dataset_path + '/test.txt', 'w')
    file_train = open(dataset_path + '/train.txt', 'w')
    file_val = open(dataset_path + '/val.txt', 'w')

    for i in range(len(total_xml)):
        line = prefix + total_xml[i][:-4] + suffix + '\n'
        if i in trainval_idx:
            # file_trainval.write(line)
            if i in val_idx:
                file_val.write(line)
            else:
                file_train.write(line)
        else:
            file_test.write(line)

    # file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


if __name__ == '__main__':
    # 传入命令行参数
    opt = parse_args()
    xml_path, dataset_path = opt.xml_path, opt.dataset_path
    # 写入txt文件
    total_xml, trainval_idx, train_idx = generate_index(xml_path, train=0.8, val=0.1)
    generate_txt(dataset_path,
                 total_xml, trainval_idx, train_idx,
                 prefix='E:/Projects/#Project_Python/datasets/face-mask-detection/images/',
                 suffix='.png')
