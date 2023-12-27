import os
import random
import torch
from d2l import torch as d2l

d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def _read_wiki(data_dir):
    """读数据，预处理

    Parameters
    ----------
    data_dir : _type_
        文件路径
    Returns
    -------
    2d list
       元素是list，是每一行文本，每一行每句话分割开
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

# nsp任务
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

# 合并句子对
# 添加segmnt：用0和1区分不同句子
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """从语料库的每一行（段落）中准备nsp任务数据

    Parameters
    ----------
    paragraph : 2d list
        语料库中每一段落数据，可能有多个句子，每个元素代表一个句子，粒度是单词级别token
    paragraphs : 3d list
        语料库，里面元素是一个段落文本，就是paragraph
    vocab : _type_
        词表，提供token和索引的双向映射
    max_len : int
        两个句子的最大长度

    Returns
    -------
    nsp_data_from_paragraph ：list
        当前段落里，所有两两相邻句子组成的句子对数据，里面的元素是一个tuple
        tuple里的数据是：（token组成的句子对， 标记不同句子的segment（元素是0和1）， 两个句子是否相邻的标签）
    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

# mlm任务
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """对一个输入句子对进行掩码处理
        1. 提供句子对中所有可以被掩码token的位置列表，其实就是出去特殊token后的整个句子对的所有位置
        2. 打乱候选位置列表，然后前15%的位置会被mask处理
        3.总共有三种mask方法，不同的概率，方法是产生一个随机值来代表概率

    Parameters
    ----------
    tokens : list
        token组成的句子对
    candidate_pred_positions : list
        可能会被mask的所有token的位置列表
        特殊标记'<cls>', '<sep>'不参与mask
    num_mlm_preds : int
        需要mask的数量
    vocab : _type_
        词表

    Returns
    -------
    mlm_input_tokens : list
        mask处理以后的token组成的句子对
    pred_positions_and_labels : list
        里面的元素是tuple：记录哪些token被mask
        （被掩码过的token的位置，正确token）
    """
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]

    pred_positions_and_labels = []
    #===========先打乱可能mask的token位置列表，排在前面15%的会被mask========================
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)

        #更改        
        mlm_input_tokens[mlm_pred_position] = masked_token
        #里面的元素是tuple：（正确token的位置，正确token）
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """对一个输入句子对进行掩码处理

    Parameters
    ----------
    tokens : list
        token组成的句子对
    vocab : _type_
        词表

    Returns
    -------
    vocab[mlm_input_tokens] : list
        mask处理以后的token的索引组成的句子对
    pred_positions : list
        里面的元素是被掩码过的token的位置
    vocab[mlm_pred_labels] : list
    记录哪些token被mask
    里面的元素是被掩码替换的正确token的索引
    """
    #可能会被mask的token位置列表
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)

    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        #填充
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        
        #=========用0填充==================
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))

        #这里的填充纯粹是考虑可能是计算掩码token数量四舍五入的误差导致的
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
        
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        #
        # 输入paragraphs[i]是list，代表一个段落，以句子为单位分割，元素是一个句子，里面是字符串
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表

        #分词，
        # paragraphs是3d 列表
        # 第一层元素是每个段落，还是list，里面有多个句子
        # 第二层元素是每个句子，还是list，里面有很多token
        # 第三层元素是token
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        
        #paragraphs铺平为2d列表，里面每个list是一个句子，以及分词过了
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        
        #创建词表
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        
        # 获取下一句子预测任务的数据
        #example里的元素是tuple：（token组成的句子对， 标记不同句子的segment（元素是0和1）， 两个句子是否相邻的标签）
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
            
        # 获取遮蔽语言模型任务的数据
        #example里的元素是tuple：
        #（mask处理以后的token的索引组成的句子对， 被掩码过的token的位置列表，被掩码替换的正确token的索引列表，
        #    标记不同句子的segment（元素是0和1）， 两个句子是否相邻的标签）
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
    
# 加载WikiText-2数据集
def load_data_wiki(batch_size, max_len):
    
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    #paragraphs是一个2d列表，每一行文本以句子为单位分割
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab

if __name__=="__main__":
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
        mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
            pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
            nsp_y.shape)
        break