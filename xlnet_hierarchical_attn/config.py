# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/6/20 13:37
Desc:
"""

project_dir = '/home/xijian/pycharm_projects/document-level-classification/'
data_base_dir = project_dir + 'data/thucnews/'

xlnet_model_dir = '../pretrained_models/chinese-xlnet-base/'

save_dir = './save/20210423/'
imgs_dir = './imgs/20210423/'

feature_extract = True # xlnet是否仅作为特征提取器，如果为否，则xlnet也参与训练，进行微调
train_from_scrach = True # 是否重头开始训练模型
last_new_checkpoint = 'epoch011_valacc0.971_ckpt.tar'

labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for i,l in enumerate(labels)}

LR = 5e-4 # 0.0005
EPOCHS = 20

doc_maxlen = 600 # 每个句子最大长度
segment_len = 150 # 段长
overlap = 50
num_classes = len(labels)

batch_size = 256