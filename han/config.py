# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/3/16 16:13
Desc:
"""

project_dir = '/home/xijian/pycharm_projects/document-level-classification/'
data_base_dir = project_dir + 'data/thucnews/'

save_dir = './save/20210316/'
imgs_dir = './imgs/20210316/'
stopwords_file = project_dir + 'data/zh_data/stopwords.txt'
vocab_path = 'tokenizer/vocab.pkl'

labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for i,l in enumerate(labels)}

LR = 1e-2
EPOCHS = 15

total_words = 6000 # 仅考虑频率最高的6000个词
doc_maxlen = 60 # 每个句子最大长度
sent_maxlen = 15
embedding_dim = 100

num_classes = len(labels)
hidden_size = 64

pad_token = '<pad>'
pad_id = 1

batch_size = 512