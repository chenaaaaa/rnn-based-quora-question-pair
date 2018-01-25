#coding=utf-8
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import word2vec
import re
from multiprocessing import Pool

##dataset 是一个类
##包含下面的成员函数

## __init__
## next_batch
## clean_str

class dataset(object):
    def __init__(self,s1,s2,label,seqlen1,seqlen2):
        self.index_in_epoch = 0
        self.s1 = s1
        self.s2 = s2
        self.label = label
	self.seqlen1 = seqlen1
	self.seqlen2 = seqlen2
##self.example_nums 是总共样本的个数
        self.example_nums = len(label)
        self.epochs_completed = 0

    def next_batch(self,batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.example_nums:
	    #print "read next epoch!"
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.label = self.label[perm]
	    self.seqlen1 = np.array(self.seqlen1)
	    self.seqlen2 = np.array(self.seqlen2)

	    self.seqlen1 = self.seqlen1[perm]
	    self.seqlen2 = self.seqlen2[perm]
	    #print "self.s1.head is", self.s1[:1]
	    #print "self.s2.head is", self.s2[:1]
	    #print "self.seqlen1.head is",self.seqlen1[:1]
	    #print "self.seqlen2.head is",self.seqlen2[:1]
            #print "self.s1.second is", self.s1[1:2]
            #print "self.s2.second is", self.s2[1:2]
            #print "self.seqlen1.second is",self.seqlen1[1:2]
            #print "self.seqlen2.second is",self.seqlen2[1:2]



            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.example_nums
        end = self.index_in_epoch
	print "call next_batch @ start = ", start, "th example , end = ", end-1,"th example"
	ratio = sum(np.array(self.label[start:end]))/batch_size *100
	print "postive/all ratio: ", ratio
        return np.array(self.s1[start:end]),\
	       np.array(self.s2[start:end]),\
	       np.array(self.label[start:end]),\
	       np.array(self.seqlen1[start:end]),\
	       np.array(self.seqlen2[start:end])

def clean_str(string):
    #
    #对句子相似度任务进行字符清洗
    #
#print "before clean_str, string = ",string
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
#   print "after clear, string = ", string
    return string.strip().lower()

def padding_sentence(s1, s2):
    #
    # 得到句子s1,s2以后，很直观地想法就是先找出数据集中的最大句子长度，
    # 然后用<unk>对句子进行填充
    #
    s1_length_max = max([len(s) for s in s1])
    s2_length_max = max([len(s) for s in s2])
    sentence_length = max(s1_length_max, s2_length_max)
    sentence_num = s1.shape[0]
    s1_padding = np.zeros([sentence_num, sentence_length], dtype=int)
    s2_padding = np.zeros([sentence_num, sentence_length], dtype=int)

    for i, s in enumerate(s1):
        s1_padding[i][:len(s)] = s

    for i, s in enumerate(s2):
        s2_padding[i][:len(s)] = s

    return s1_padding, s2_padding

def get_id(word):
    if word in sr_word2id:
        return sr_word2id[word]
    else:
        return sr_word2id['<unk>']

def seq2id(seq):

    seq = clean_str(seq)
#print "seq = ", seq
#print "seq=clean_str is finish"
    seq_split = seq.split(' ')
    seq_id = map(get_id, seq_split)
#print "len(seq)=", len(seq_id)
#print"seq_id = map is finish"
    return seq_id

def read_data_sets(train_dir):
    #
    # s1代表数据集的句子1
    # s2代表数据集的句子2
    # score代表相似度
    # sample_num代表数据总共有多少行
    #
    df_sick = pd.read_csv(train_dir, usecols=[3,4,5], names=['s1', 's2', 'score'], dtype={'s1':object, 's2':object, 'score':object})

    print "df_sick.shape is", df_sick.shape
    #print(df_sick.head(5))
    df_sick = df_sick.drop([0])
    s1 = df_sick.s1.values
    #print("s1.head is", s1[:5])
    #print("")
    #print("")
    #print("")

    s2 = df_sick.s2.values
    #print("s2.head is", s2[:5])
    #print("")
    #print("")
    #print("")
    score = np.asarray(map(float, df_sick.score.values), dtype=np.float32)
    #print
    #print
    #print
    sample_num = len(score)
    #print "len_socre = ", sample_num

    #print
    #print


    # 引入embedding矩阵和字典
    global sr_word2id, word_embedding
    sr_word2id, word_embedding = build_glove_dic()
    print "build_glove_dic() finish"

    # word2id, 多线程将word转成id
    p = Pool()
    s1 = np.asarray(p.map(seq2id, s1))


    seqlen1 = []
    seqlen2 = []

#    s1 = np.asarray(map(seq2id, s1))

    #print("after seq2id, s1.head is", s1[:5])
    #print "s1.len is", len(s1)

    seqlen1 = [len(x)-1 for x in s1]

    for idx , x in enumerate(s1):
	seqlen1.append(len(s1[idx]) -1 )

    #print "-------------------------"
    #print "s1.head is", s1[:5]
    #print "seqlen1.head is", seqlen1[:5]
    #print "score.head is", score[:5] 
    #print "-------------------------"
    #print "s1.tail is", s1[-1-5:]
    #print "seqlen1.tail is", seqlen1[-1-5:]
    #print "-------------------------"
    #print "score.tail is", score[-1-5:] 

    s2 = np.asarray(p.map(seq2id, s2))
    #s2 = np.asarray(map(seq2id, s2))

    seqlen2 = [(len(x)-1) for x in s2]

#    for idx , x in enumerate(s2):
#	        seqlen2.append(len(s2[idx]) -1 )	

    #print "-------------------------"
    #print "s2.head is", s2[:5]
    #print "seqlen2.head is", seqlen2[:5]
    #print "score.head is", score[:5] 
    #print "-------------------------"
    #print "s2.tail is", s2[-1-5:]
    #print "seqlen2.tail is ", seqlen2[-1-5:]
    #print "score.tail is", score[-1-5:] 
    #print "-------------------------"
    #print "s1.len is", len(s1)
    #print "s2.len is", len(s2)
    #print "score.len is", len(score)
    #print "in data_helper.py score.type is",type(score)
    #print "in data_helper.py score.len is",len(score)

#print "s2[0].type is", s2[0].type
#    p.close()
#    p.join()
#    print "p.join() finish"

    # 填充句子
    print "about to padding sentense"
    s1, s2 = padding_sentence(s1, s2)

#new_index = np.random.permutation(sample_num)
#s1 = s1[new_index]
#s2 = s2[new_index]
#score = score[new_index]

    return s1 ,s2, score, seqlen1, seqlen2

def build_glove_dic():
    # 从文件中读取 pre-trained 的 glove 文件，对应每个词的词向量
    # 需要手动对glove文件处理，在第一行加上
    # 400000 50
    # 其中400000代表共有四十万个词，每个词50维，中间为一个空格或者tab键
    # 因为word2vec提取需要这样的格式，详细代码可以点进load函数查看
    glove_path = 'glove.6B.50d.txt'
    wv = word2vec.load(glove_path)
    vocab = wv.vocab
##sr_word2id 类似于字典的作用, 所有的词都是从1开始编号的
    sr_word2id = pd.Series(range(1,len(vocab) + 1), index=vocab)
##unk的编号是0
    sr_word2id['<unk>'] = 0
    word_embedding = wv.vectors
    word_mean = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_mean, word_embedding])
## word_embedding 是从word2vec.load(glove_path).vectors来的
## 
    return sr_word2id, word_embedding

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# 如果运行该文件，执行此命令，否则略过
if __name__ == "__main__":
    read_data_sets(0)
