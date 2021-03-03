#coding=utf-8
'''
Created on 2020-9-4

@author: Yoga
'''

import pandas as pd
import re
from matplotlib import pyplot as plt

filepath = 'cnews.train.txt'
 
# 1.处理训练数据  数据格式：content    label
df_data = pd.read_csv(filepath, encoding='UTF-8', sep = '\t', names=['label', 'content'], index_col=False)
df_data = df_data.dropna()
print(df_data.head())

# 文本长度（所含字数）
df_data['content_len'] = df_data.content.apply(lambda x: len(x))
print(df_data['content_len'].describe())#最大文档长度即最长文档字数可定在500字

# 给文档分句，返回句子列表，每个句子也进行了分词，词与词之间用空格隔开
def seg_sentences(doc):
    #sents = re.split(r'，|。|！|？|：|；|,|!|\?|:|;', doc)
    sents = re.split(r'，|。|！|？|；|!|\?|;', doc)
    sents = [s.strip() for s in sents if len(s.strip())>0]
    return sents

# 分句
df_data['sentences'] = df_data.content.apply(lambda x: seg_sentences(x))
print(df_data.iloc[0]['sentences'])

# 句子数
df_data['sentences_num'] = df_data.sentences.apply(lambda x: len(x))
print(df_data['sentences_num'].describe())
df_data.sentences_num.hist(bins=100) # 最大句子数可定在60个句子
plt.show()


# 每个句子所含字数
words_per_sent = []
for sents in df_data['sentences']:
    for s in sents:
        words_per_sent.append(len(s))
df_sent_words = pd.DataFrame(words_per_sent, columns=['words_num'])
print(df_sent_words.head())
# print(df_sent_words['words_num'].value_counts())
print(df_sent_words['words_num'].describe()) # 最大句子长度（最长字数）可定在15个字
print('{}个句子'.format(len(words_per_sent)))
# df_sent_words.words_num.hist(bins=10)
plt.show()