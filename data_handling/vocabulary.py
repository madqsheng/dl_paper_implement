import collections

class Vocab: 
    """token和索引互相转换的类

    token是单词级别、字母级别取决与参数tokens
    """
    def __init__(self, tokens, min_freq=0, reserved_tokens=['<pad>', '<bos>', '<eos>']):
        """初始化

        tokens是文本集合以token级别进行分割后的list，元素是token，可以是单词，字母
        对tokens里的数据进行遍历，如果是2D列表，就会先展开再遍历。 

        '<unk>'是已经预留的token，索引为0，代表位置token   

        关键属性：
        self.idx_to_token：list，根据频数token从大到小
        self.token_to_idx：dict，{token:index}
        self._token_freqs：dict，对tokens里token的频数统计，从大到小排序的结果

        Parameters
        ----------
        tokens : list
            1D列表或2D列表,文本按照token级别分割后的list
        min_freq : int, optional
            频数阈值，低于该频数的token忽略, by default 0
        reserved_tokens : list, optional
            预留的token，常见有：['<pad>', '<bos>', '<eos>']，索引排在前列，内部, by default ['<pad>', '<bos>', '<eos>']
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 对tokens里的元素进行频数统计，输出是dict
        counter = count_corpus(tokens)
        #按照频率从大到小排序
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        #索引和token的双向映射
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """token到索引的映射

        Parameters
        ----------
        tokens : (str ,list,tuple)
            str类型代表一个token
            list/tuple 代表多个token

        Returns
        -------
        (int,list)
            tokens是str，则返回一个索引int，未知则返回0
            tokens是list/tuple,则返回一个list，里面是索引
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    #输入索引，给出token
    def to_tokens(self, indices):
        """索引到token的映射

        Parameters
        ----------
        indices : (int,list,tuple)
            单个索引或者多个索引

        Returns
        -------
        (str,list)
            tokens是int，则返回一个token,类型是str
            tokens是list/tuple,则返回一个list，里面是token
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

if __name__=="__main__":
    #源数据的vocab类
    src_vocab = Vocab(source, min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(len(src_vocab))

    tg_vocab = Vocab(target, min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(len(tg_vocab))