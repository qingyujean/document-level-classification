# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/6/20 15:47
Desc:
"""
import time
import pandas as pd

import torch
from transformers import BertTokenizer

import sys
sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from bert_overlap_split_bilstm.config import *

tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

# 读取数据  数据格式：content    label
def read_data(filepath, tokenizer):
    df_data = pd.read_csv(filepath, encoding='UTF-8', sep='\t', names=['label', 'content'], index_col=False)
    df_data = df_data.dropna()

    x_data, y_data = df_data['content'], df_data['label']
    print('*' * 27, x_data.shape, len(x_data[0]), y_data.shape)  # (50000,) 746 (50000,)

    x_data = bert_encode(x_data, tokenizer)

    y_data = [label2id[y] for y in y_data]
    y_data = torch.tensor(y_data, dtype=torch.long)

    return x_data, y_data



def bert_encode(texts, tokenizer):
    starttime = time.time()
    print('*'*27, 'start encoding...')
    inputs = tokenizer.batch_encode_plus(texts, return_tensors='pt', add_special_tokens=True,
                                         max_length=doc_maxlen, # max_length 已经把句子最后的<sep><cls>算进去了，所以其实最长句子是max_length-2
                                         padding='longest',  # 默认是False  向batch里最长的句子补齐
                                         truncation='longest_first')  # 按max_length截断
                                         # padding=False,
                                         # truncation=False) # 不截断和pad  后面组batch时再pad 以提高性能 这种方法不行  因为如果使用batch_encode_plus必须有pad=True？
    # print(inputs) # input_ids token_type_ids attention_mask
    endtime = time.time()
    print('*'*27, 'data to ids finished...')
    print('*'*27, 'and it costs {} min {:.2f} s'.format(int((endtime-starttime)//60), (endtime-starttime)%60))
    return inputs



def load_data(filepath, tokenizer, shuffle=False):
    inputs, y_data = read_data(filepath, tokenizer)

    # TensorDataset 参数只能是tensor
    inp_dset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'],
                                              y_data)
    inp_dloader = torch.utils.data.DataLoader(inp_dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=2)
    return inp_dloader


if __name__ == '__main__':
    """
    x_data_val, y_data_val = read_data(data_base_dir + 'cnews.val.txt', tokenizer)
    print(x_data_val.keys(), x_data_val['input_ids'][:10])
    print(y_data_val[:10])
    """

    data_loader = load_data(data_base_dir + 'cnews.val.txt', tokenizer)
    x_0, x_1, x_2, y = next(iter(data_loader))
    print('sample:', 'x:', x_0[:10], x_0.shape, x_1.shape, x_2.shape, 'y:', y.shape[:10]) # [b, max_doclen=600], y: [b]
