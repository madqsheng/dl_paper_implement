import torch
from torch.utils.data import Dataset,DataLoader
from vocabulary import Vocab
import random

def tokenize(lines, token='word'):
    """将1D的list差分为2D的list

    对每行文本再划分，级别：
    1.单词级别
    2.字母级别

    Parameters
    ----------
    lines : list 1d
        元素是str，每行文本
    token : str, 'word'|'char'
        对每行文本划分的级别, by default 'word'

    Returns
    -------
    list，2d
        还是所有文本，一层list元素是每行文本，二层元素是每行的token
    """
    
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def seq_data_iter_random(corpus, batch_size, num_steps):
    """语料库生成小批量序列办法：随机取样

    来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻

    Parameters
    ----------
    corpus : list，1D
        语料库，完成文本数据，元素是token或者索引
    batch_size : int
        批量大小
    num_steps : int_
        每个序列的长度

    Returns
    迭代器
    -------
        tensor,形状：(batch_size,num_steps)
            单个batch的序列数据

        ------
        tensor,形状：(batch_size,num_steps)
            以上单个batch序列向右偏移一个时间步的序列
    """
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]

    # 序列数目
    # 减去1，是因为我们需要考虑标签
    # 所谓语言模型的标签其实是x向右偏移一个时间步
    num_subseqs = (len(corpus) - 1) // num_steps

    # 每个序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]

        # #一个batch数据：2维:(batch_size,num_steps)
        X = [data(j) for j in initial_indices_per_batch] 
        Y = [data(j + 1) for j in initial_indices_per_batch] 
        yield torch.tensor(X), torch.tensor(Y)

"""使用顺序分区生成一个小批量子序列"""
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """语料库生成小批量序列办法：顺序采样

    相同batch里的子序列不是相邻的。
    相邻batch里同位置子序列是相邻的

    Parameters
    ----------
    corpus : list，1D
        语料库，完成文本数据，元素是token或者索引
    batch_size : int
        批量大小
    num_steps : int_
        每个序列的长度

    Returns
    迭代器
    -------
        tensor,形状：(batch_size,num_steps)
            单个batch的序列数据

        ------
        tensor,形状：(batch_size,num_steps)
            以上单个batch序列向右偏移一个时间步的序列
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size

    #序列是相邻的
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

def load_corpus_time_machine(tokens,max_tokens=-1):
    """输入tokens，返回语料库和词表

    Parameters
    ----------
    tokens : 2D list
        文本token
    max_tokens : int, optional
        语料库最长tokens数目

    Returns
    -------
    corpus ：list 1d
        语料库，预案是索引
    vocab ： Vocab类
        词表
    """
    vocab = Vocab(tokens,reserved_tokens=[])
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab



class SeqDataLoader:
    def __init__(self, tokens,batch_size, num_steps, max_tokens=10000, use_random_iter=False):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(tokens,max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(tokens,batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        tokens,batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

if __name__=="__main__":
    import os
    import re

    data_dir='./../data'
    with open(os.path.join(data_dir, 'timemachine.txt'), 'r',encoding='utf-8') as f:
        #一次性多行读取，list，每个元素是一行str
        lines = f.readlines()
        
        lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    #字符串
    print(type(lines))
    print(lines[:75])

    char_tokens = tokenize(lines,'char')
    for i in range(11):
        print(char_tokens[i])

    corpus, vocab = load_corpus_time_machine(char_tokens)
    len(corpus), len(vocab)