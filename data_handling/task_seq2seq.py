import torch
from torch.utils.data import Dataset,DataLoader
from vocabulary import Vocab

def preprocess_nmt(text):
    """对文本数据预处理，单词之间用空格分开：

    1. 特殊空格符号，统一空格
    2. 统一小写
    3. 单词和标点符号之间加上空格
    4. 确定token级别是单词，不同token之间用空格分开

    Parameters
    ----------
    text : str
        文本字符串

    Returns
    -------
    str
        文本字符串
    """

    # char是字符，prev_char是前一个字符
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在单词和标点符号之间插入空格
    #遍历字符串
    #如果当前字符是标点符号，并且上一个字符不是空格。那么在当前字符前面加空格
    #其他情况一律保留当前字符
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """对文本字符串分割，三个分割级别：

    1.换行符：分割不同序列
    2.制表符：分割源语言和目标语言
    3.空格：分割不同token

    Parameters
    ----------
    text : str
        文本字符串，里面三个分隔符：换行符，制表符，空格
    num_examples : int, optional
        从text提取序列数量, by default None

    Returns
    -------
    source : 2D list
        源语言所有token序列，内部是1D list，里面元素是token,包括标点符号
    target : 2D list
        目标语言所有token序列，内部是1D list，里面元素是token,包括标点符号
    """

    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


class SequenceDataset(Dataset):
    def __init__(self,tokens,num_steps=10,min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>']):
        """初始化所有序列tokens

        1.内部元素token换为索引
        2.在每个序列后面添加ending_token
        3.对每个序列进行截断和填充处理，序列固定长度：num_steps，填充token：padding_token

        Parameters
        ----------
        tokens : 2D list
            所有序列tokens，内部是1D list，里面元素是token,包括标点符号
        num_steps : int, optional
            对每个序列进行截断填充处理，长度参数, by default 10
        min_freq : int, optional
            token和索引互相转换的类，对tokens中所有token进行统计，频数低于min_freq忽略, by default 2
        reserved_tokens : list, optional
            token和索引互相转换的类，预留的token，特殊用处，索引排在前列, by default ['<pad>', '<bos>', '<eos>']
        """
        self.tokens = tokens
        self.vocab = Vocab(self.tokens, min_freq=min_freq,reserved_tokens =reserved_tokens)
        self.truncate_pad_tensor, self.valid_len, = self.truncate_pad(num_steps)
    
    def __len__(self):
        return self.truncate_pad_tensor.shape[0]
    
    def __getitem__(self,index):
        return (self.truncate_pad_tensor[index], self.valid_len[index])
    
    def truncate_pad(self,num_steps,ending_token='<eos>', padding_token='<pad>'):
        """对所有序列self.tokens中每个序列：
        1.后面加上ending_token
        2.截断填充处理
        3.转换tensor数据类型

        num_steps是固定长度
        超过长度截断
        没超过长度，用padding_token填充
        ----------
        num_steps : int
            每个序列固定长度
        ending_token : str, optional
            每个序列后面添加的token, by default '<eos>'
        padding_token : str, optional
            每个序列填充的token, by default '<pad>'

        Returns
        -------
        truncate_pad_tensor ： 2D tensor
            处理完的所有序列索引
        valid_len ： 1D tensor
            所有序列的有效长度
        """
        sequences_idx = [self.vocab[sequence] for sequence in self.tokens]
        #每个序列后面加一个'<eos>'索引
        sequences_idx = [idx + [self.vocab[ending_token]] for idx in sequences_idx]
        # 每个序列截断填充处理
        truncate_pad=(
            [sequence_idx[:num_steps] if len(sequence_idx)>num_steps 
            else sequence_idx + [self.vocab[padding_token]] * (num_steps - len(sequence_idx)) 
            for sequence_idx in sequences_idx])

        # 形状：(s,num_steps)
        # 其中s是序列数目，num_steps是固定的序列长度
        truncate_pad_tensor = torch.tensor(truncate_pad)
        # 有效长度
        valid_len = (truncate_pad_tensor != self.vocab['<pad>']).type(torch.int32).sum(1)
        return (truncate_pad_tensor, valid_len)

class Seq2seqDataset(Dataset):
    def __init__(self,source_dataset,target_dataset):
        """构建seq2seq训练数据类型

        Parameters
        ----------
        source_dataset : SequenceDataset
            源语言的SequenceDataset实例
        target_dataset : SequenceDataset
            目标的SequenceDataset实例
        """
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        assert len(self.source_dataset)==len(self.target_dataset)
        return (len(self.source_dataset))
    
    def __getitem__(self,index):
        """源语言和目标语言序列和长度一一对应

        Parameters
        ----------
        index : int
            索引

        Returns
        -------
        seq2seq数据 : tuple
            源语言某个序列的tensor，有效长度，目标语言某个序列的tensor，有效长度
        """
        return (self.source_dataset[index][0],self.source_dataset[index][1],
        self.target_dataset[index][0],self.target_dataset[index][1])

if __name__=="__main__":
    import os

    data_dir='./../data/fra-eng'
    with open(os.path.join(data_dir, 'fra.txt'), 'r',encoding='utf-8') as f:
        raw_text= f.read()
    #字符串
    print(type(raw_text))
    print(raw_text[:75])

    text = preprocess_nmt(raw_text)
    print(type(text))
    print(text[:80])


    #source, target都是双层list，里面一层的list是不定长度的序列
    #token是单词，包括标点符号
    source_tokens, target_tokens = tokenize_nmt(text)
    source_tokens[:6], target_tokens[:6]

    source_dataset = SequenceDataset(source_tokens)
    target_dataset = SequenceDataset(target_tokens)

    seq2seq_dataset = Seq2seqDataset(source_dataset,target_dataset)
    seq2seq_dataset[0]

    data_iter= DataLoader(seq2seq_dataset, batch_size=2, shuffle=True)
    for X, X_valid_len, Y, Y_valid_len in data_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break