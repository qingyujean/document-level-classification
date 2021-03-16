# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/1/29 10:58
Desc:
"""
project_dir = '/home/xijian/pycharm_projects/document-level-classification/'
data_base_dir = project_dir + 'data/thucnews/'

save_dir = './save/20210303/'
imgs_dir = './imgs/20210303/'
stopwords_file = project_dir + 'data/zh_data/stopwords.txt'
counter_path = 'tokenizer/counter.pkl'

feature_extract = True # xlnet是否仅作为特征提取器，如果为否，则xlnet也参与训练，进行微调

labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for i,l in enumerate(labels)}

LR = 1e-3
EPOCHS = 30

total_words = 10000 # 仅考虑频率最高的10000个词
doc_maxlen = 500 # 每个句子最大长度
num_classes = len(labels)
net_depth = 15 # 15

pad_token = '<pad>'
pad_id = 1
embedding_dim = 100
batch_size = 512 # 256

