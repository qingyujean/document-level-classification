# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/6/20 15:46
Desc:
"""
import torch
from transformers import BertTokenizer

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
            confusion_matrix, classification_report
import time
import sys

sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from bert_overlap_split_bilstm.config import *
from bert_overlap_split_bilstm.prepare_data import load_data
from bert_overlap_split_bilstm.train import BertLSTMWithOverlap, printbar

tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

ngpu = 4 # 4

use_cuda = torch.cuda.is_available() # 检测是否有可用的gpu
device = torch.device("cuda:0" if (use_cuda and ngpu>0) else "cpu")
print('*'*8, 'device:', device)


# checkpoint = save_dir + 'epoch018_valacc0.948_ckpt.tar'
checkpoint = save_dir + last_new_checkpoint


@torch.no_grad()
def eval_step(model, inps, labs):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.eval()  # 设置eval mode

    # forward
    logits = model(input_ids, token_type_ids, attention_mask)
    pred = torch.argmax(logits, dim=-1)

    return pred, labs


def evaluate(model, test_dloader):
    starttime = time.time()
    print('*' * 27, 'start evaluating...')
    printbar()
    preds, labels = [], []
    for step, (inp_ids, type_ids, att_mask, labs) in enumerate(tqdm(test_dloader), start=1):
        inps = (inp_ids, type_ids, att_mask)
        pred, labs = eval_step(model, inps, labs)
        preds.append(pred)
        labels.append(labs)

    y_true = torch.cat(labels, dim=0)
    y_pred = torch.cat(preds, dim=0)
    endtime = time.time()
    print('evaluating costs: {:.2f}s'.format(endtime - starttime))
    return y_true.cpu(), y_pred.cpu()


def get_metrics(y_true, y_pred):
    if num_classes == 2:
        print('*'*27, 'precision_score:', precision_score(y_true, y_pred, pos_label=1))
        print('*'*27, 'recall_score:', recall_score(y_true, y_pred, pos_label=1))
        print('*'*27, 'f1_score:', f1_score(y_true, y_pred, pos_label=1))
    else:
        average = 'weighted'
        print('*'*27, average+'_precision_score:{:.3f}'.format(precision_score(y_true, y_pred, average=average)))
        print('*'*27, average+'_recall_score:{:.3}'.format(recall_score(y_true, y_pred, average=average)))
        print('*'*27, average+'_f1_score:{:.3f}'.format(f1_score(y_true, y_pred, average=average)))

    print('*'*27, 'accuracy:{:.3f}'.format(accuracy_score(y_true, y_pred)))
    print('*'*27, 'confusion_matrix:\n', confusion_matrix(y_true, y_pred))
    print('*'*27, 'classification_report:\n', classification_report(y_true, y_pred))



if __name__ == '__main__':
    test_dloader = load_data(data_base_dir + 'cnews.test.txt', tokenizer)

    sample_batch = next(iter(test_dloader))
    print('*' * 27, 'sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[0].dtype,
          sample_batch[1].size(), sample_batch[1].dtype,
          sample_batch[2].size(), sample_batch[2].dtype,
          sample_batch[3].size(), sample_batch[3].dtype)

    model = BertLSTMWithOverlap(bert_model_dir, num_classes, segment_len=segment_len, overlap=overlap)
    model = model.to(device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1,2,3]

    print('*' * 27, 'Loading model weights...')
    # ckpt = torch.load(checkpoint, map_location=device)  # dict  save在CPU 加载到GPU
    ckpt = torch.load(checkpoint)  # dict  save在GPU 加载到 GPU
    model_sd = ckpt['net']
    if device.type == 'cuda' and ngpu > 1:
        model.module.load_state_dict(model_sd)
    else:
        model.load_state_dict(model_sd)
    print('*' * 27, 'Model loaded success!')

    y_true, y_pred = evaluate(model, test_dloader)
    get_metrics(y_true, y_pred)