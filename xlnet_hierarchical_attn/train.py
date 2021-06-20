# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/6/20 13:37
Desc:
"""
import torch
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import sys
import os

sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from xlnet_hierarchical_attn.config import *
from xlnet_hierarchical_attn.prepare_data import load_data

tokenizer = XLNetTokenizer.from_pretrained(xlnet_model_dir+'spiece.model')

ngpu = 4

use_cuda = torch.cuda.is_available() # 检测是否有可用的gpu
device = torch.device("cuda:0" if (use_cuda and ngpu>0) else "cpu")
print('*'*8, 'device:', device)


# 设置损失函数和评价指标
loss_func = torch.nn.CrossEntropyLoss()
metric_func = lambda y_pred, y_true: accuracy_score(y_true, y_pred)
metric_name = 'acc'
df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])


# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)


class NyAttentioin(torch.nn.Module):
    def __init__(self, hidden_size, attensize_size):
        super(NyAttentioin, self).__init__()

        self.attn = torch.nn.Linear(hidden_size, attensize_size)
        self.ctx = torch.nn.Linear(attensize_size, 1, bias=False)

    # inputs: [b, seq_len, hidden_size]
    def forward(self, inputs):
        u = self.attn(inputs).tanh() # [b, seq_len, hidden_size]=>[b, seq_len, attention_size]
        scores = self.ctx(u) # [b, seq_len, attention_size]=>[b, seq_len, 1]
        attn_weights = F.softmax(scores, dim=1) # [b, seq_len, 1]

        out = torch.bmm(inputs.transpose(1, 2), attn_weights) # [b, seq_len, hidden_size]=>[b, hidden_size, seq_len]x[b, seq_len, 1]=>[b, hidden_size, 1]
        return torch.squeeze(out, dim=-1) # [b, hidden_size, 1]=>[b, hidden_size]


class MyXLNetModel(torch.nn.Module):
    def __init__(self, pretrained_model_dir, num_classes, segment_len=150, dropout_p=0.5):
        super(MyXLNetModel, self).__init__()

        self.seg_len = segment_len

        self.config = XLNetConfig.from_json_file(pretrained_model_dir + 'config.json')
        self.config.mem_len = 150  # enable the memory #
        self.xlnet = XLNetModel.from_pretrained(pretrained_model_dir, config=self.config)

        if feature_extract:
            for p in self.xlnet.parameters():  # 迁移学习：xlnet作为特征提取器
                p.requires_grad = False

        d_model = self.config.hidden_size  # 768
        self.attention_layer1 = NyAttentioin(d_model, d_model // 2)
        self.attention_layer2 = NyAttentioin(d_model, d_model // 2)

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc = torch.nn.Linear(d_model, num_classes)


    def get_segments_from_one_batch(self, input_ids, token_type_ids, attention_mask):
        doc_len = input_ids.shape[1]
        q, r = divmod(doc_len, self.seg_len)  # 得到商和余数
        if r > 0:
            split_chunks = [self.seg_len] * q + [r]
        else:  # r==0
            split_chunks = [self.seg_len] * q

        # chunk 分块：沿着第1维进行分块， 因此分割成seg_num个Tensor， 当不能整除时， 最后一个的维数会小于前面的
        # split 分块：split_size_or_sections指定每一块的长度或者指定一个块长度list
        # 这里适合split
        input_ids = torch.split(input_ids, split_size_or_sections=split_chunks, dim=1)
        token_type_ids = torch.split(token_type_ids, split_size_or_sections=split_chunks, dim=1)
        attention_mask = torch.split(attention_mask, split_size_or_sections=split_chunks, dim=1)

        split_inputs = [{'input_ids': input_ids[seg_i], 'token_type_ids': token_type_ids[seg_i],
                         'attention_mask': attention_mask[seg_i]}
                        for seg_i in range(len(input_ids))]
        return split_inputs


    def forward(self, input_ids, token_type_ids, attention_mask):
        split_inputs = self.get_segments_from_one_batch(input_ids, token_type_ids, attention_mask)

        lower_intra_seg_repr = []  # 段内表征（低阶表征）
        mems = None
        for idx, seg_inp in enumerate(split_inputs):
            last_hidden, mems = self.xlnet(**seg_inp, mems=mems)  # last_hidden: [b, 150, 768] mems: list: [150, b, 768]
            lower_intra_seg_repr.append(self.attention_layer1(last_hidden))  # attention_layer1: [b, 150, 768]=>[b, 768]

        lower_intra_seg_repr = [torch.unsqueeze(seg, dim=1) for seg in lower_intra_seg_repr]  # list: [b, 768]=>[b, 1, 768]
        lower_intra_seg_repr = torch.cat(lower_intra_seg_repr, dim=1)  # [b, 1, 768]=>[b, num_seg, 768]

        higher_inter_seg_repr = self.attention_layer2(lower_intra_seg_repr)  # 段间表征（高阶表征）[b, num_seg, 768]=>[b, 768]

        logits = self.fc(self.dropout(higher_inter_seg_repr))  # [b, 768]=>[b, num_classes]
        return logits



def train_step(model, inps, labs, optimizer):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.train()  # 设置train mode
    optimizer.zero_grad()  # 梯度清零

    # forward
    logits = model(input_ids, token_type_ids, attention_mask)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)
    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy()) # 返回的是tensor还是标量？

    # backward
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    return loss.item(), metric.item()


def validate_step(model, inps, labs):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.eval()  # 设置eval mode

    # forward
    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = loss_func(logits, labs)

        pred = torch.argmax(logits, dim=-1)
        metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())  # 返回的是tensor还是标量？

    return loss.item(), metric.item()


def train_model(model, train_dloader, val_dloader, optimizer, scheduler_1r=None, init_epoch=0, num_epochs=10, print_every=150):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_metric = 0.
    for epoch in range(init_epoch+1, init_epoch+num_epochs+1):
        # 训练
        loss_sum, metric_sum = 0., 0.
        for step, (inp_ids, type_ids, att_mask, labs) in enumerate(train_dloader, start=1):
            inps = (inp_ids, type_ids, att_mask)
            loss, metric = train_step(model, inps, labs, optimizer)
            loss_sum += loss
            metric_sum += metric

            # 打印batch级别日志
            if step % print_every == 0:
                print('*'*27, f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

        # 验证 一个epoch的train结束，做一次验证
        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inp_ids, type_ids, att_mask, labs) in enumerate(val_dloader, start=1):
            inps = (inp_ids, type_ids, att_mask)
            val_loss, val_metric = validate_step(model, inps, labs)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        if scheduler_1r:
            scheduler_1r.step()

        # 记录和收集 1个epoch的训练和验证信息
        # columns=['epoch', 'loss', metric_name, 'val_loss', 'val_'+metric_name]
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        # 打印epoch级别日志
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
               record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        # 保存最佳模型参数
        current_metric_avg = val_metric_sum/val_step
        if current_metric_avg > best_metric:
            best_metric = current_metric_avg
            checkpoint = save_dir + f'epoch{epoch:03d}_valacc{current_metric_avg:.3f}_ckpt.tar'
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())
            # 保存
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)


    endtime = time.time()
    time_elapsed = endtime - starttime
    print('*' * 27, 'training finished...')
    print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                               int((time_elapsed % 3600) // 60),
                                                               (time_elapsed % 3600) % 60))

    print('Best val Acc: {:4f}'.format(best_metric))
    return df_history


# 绘制训练曲线
def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])

    plt.savefig(imgs_dir + metric + '.png')  # 保存图片
    plt.show()



if __name__ == '__main__':
    train_dloader = load_data(data_base_dir + 'cnews.train.txt', tokenizer, shuffle=True)
    val_dloader = load_data(data_base_dir + 'cnews.val.txt', tokenizer)

    sample_batch = next(iter(train_dloader))
    print('sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[1].size(), sample_batch[2].size(),
          sample_batch[0].dtype, sample_batch[3].size(), sample_batch[3].dtype)  # 4   [b, doc_maxlen] int64


    model = MyXLNetModel(xlnet_model_dir, num_classes, segment_len=segment_len)
    model = model.to(device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1]

    # ===================================================================================================== new add
    # 不从头开始训练，而是从最新的那个checkpoint开始训练，或者可以收到指定从某个checkpoint开始训练
    if train_from_scrach is False and len(os.listdir(os.getcwd() + '/' + save_dir)) > 0:
        print('*' * 27, 'Loading model weights...')
        ckpt = torch.load(save_dir + last_new_checkpoint)  # dict  save在GPU 加载到 GPU
        init_epoch = int(last_new_checkpoint.split('_')[0][-3:])
        print('*' * 27, 'init_epoch=', init_epoch)
        model_sd = ckpt['net']
        if device.type == 'cuda' and ngpu > 1:
            model.module.load_state_dict(model_sd)
        else:
            model.load_state_dict(model_sd)
        print('*' * 27, 'Model loaded success!')
    # =====================================================================================================

    model.eval()
    sample_out = model(sample_batch[0], sample_batch[1], sample_batch[2])
    print('*' * 10, 'sample_out:', sample_out.shape)  # [b, 10]

    # 设置优化器
    print('Params to learn:')
    if feature_extract:  # 特征提取
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print('\t', name)
    else:  # 微调
        params_to_update = model.parameters()
        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #          print('\t', name)

    optimizer = torch.optim.Adam(params_to_update, lr=LR, weight_decay=1e-4)
    scheduler_1r = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch > EPOCHS * 0.8 else 1)

    train_model(model, train_dloader, val_dloader, optimizer, scheduler_1r,
                init_epoch=init_epoch, num_epochs=EPOCHS, print_every=50)

    plot_metric(df_history, 'loss')
    plot_metric(df_history, metric_name)