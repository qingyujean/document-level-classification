# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/3/16 16:11
Desc:
"""
import torch
import torchkeras
import torch.nn.functional as F

from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import time
import sys

sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from han.config import *
from han.prepare_data import load_data


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


class MyHAN(torch.nn.Module):
    def __init__(self, max_word_num, max_sents_num, vocab_size, hidden_size, num_classes, embedding_dim, embedding_matrix=None, dropout_p=0.5):
        super(MyHAN, self).__init__()

        self.max_word_num = max_word_num  # 15 句子所含最大词数
        self.max_sents_num = max_sents_num  # 60 文档所含最大句子数

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=pad_id)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            for p in self.embedding.parameters():
                p.requires_grad = False

        self.dropout0 = torch.nn.Dropout(dropout_p)

        # self.layernorm1 = torch.nn.LayerNorm(normalized_shape=(sent_maxlen, embedding_dim), eps=1e-6)
        # self.layernorm2 = torch.nn.LayerNorm(normalized_shape=2*hidden_size, eps=1e-6)

        self.bi_rnn1 = torch.nn.GRU(self.embedding_dim, self.hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.word_attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.word_ctx = torch.nn.Linear(self.hidden_size, 1, bias=False)

        self.bi_rnn2 = torch.nn.GRU(2 * self.hidden_size, self.hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.sent_attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.sent_ctx = torch.nn.Linear(self.hidden_size, 1, bias=False)

        self.dropout = torch.nn.Dropout(dropout_p)
        self.out = torch.nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, inputs, hidden1=None, hidden2=None):  # [b, 60, 15]
        embedded = self.dropout0(self.embedding(inputs))  # =>[b, 60, 15, 100]

        word_inputs = embedded.view(-1, embedded.size()[-2], embedded.size()[-1])  # =>[b*60, 15, embedding_dim]
        # word_inputs = self.layernorm1(word_inputs)
        self.bi_rnn1.flatten_parameters()
        """
        为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)。
        类似我们调用tensor.contiguous
        """
        word_encoder_output, hidden1 = self.bi_rnn1(word_inputs,
                                                    hidden1)  # =>[b*60,15,2*hidden_size], [b*60,2,hidden_size]
        word_attn = self.word_attn(word_encoder_output).tanh()  # =>[b*60,15,hidden_size]
        word_attn_energy = self.word_ctx(word_attn)  # =>[b*60,15,1]
        word_attn_weights = F.softmax(word_attn_energy, dim=1).transpose(1, 2)  # =>[b*60,15,1]=>[b*60,1,15]
        word_att_level_output = torch.bmm(word_attn_weights, word_encoder_output)  # =>[b*60,1,2*hidden_size]

        sent_inputs = word_att_level_output.squeeze(1).view(-1, self.max_sents_num,
                                                            2 * self.hidden_size)  # =>[b*60,2*hidden_size]=>[b,60,2*hidden_size]
        self.bi_rnn2.flatten_parameters()
        sent_encoder_output, hidden2 = self.bi_rnn2(sent_inputs, hidden2)  # =>[b,60,2*hidden_size], [b,2,hidden_size]
        sent_attn = self.sent_attn(sent_encoder_output).tanh()  # =>[b,60,hidden_size]
        sent_attn_energy = self.sent_ctx(sent_attn)  # =>[b,60,1]
        sent_attn_weights = F.softmax(sent_attn_energy, dim=1).transpose(1, 2)  # =>[b,60,1]=>[b,1,60]
        sent_att_level_output = torch.bmm(sent_attn_weights, sent_encoder_output)  # =>[b,1,2*hidden_size]

        # logits = self.out(self.dropout(self.layernorm2(sent_att_level_output.squeeze(1))))  # =>[b,2*hidden_size]=>[b,num_classes]
        logits = self.out(self.dropout(sent_att_level_output.squeeze(1)))  # =>[b,2*hidden_size]=>[b,num_classes]
        return logits  # [b,num_classes]


def train_step(model, inps, labs, optimizer):
    inps = inps.to(device)
    labs = labs.to(device)

    model.train()  # 设置train mode
    optimizer.zero_grad()  # 梯度清零

    # forward
    logits = model(inps)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)
    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy()) # 返回的是tensor还是标量？
    # print('*'*8, metric)

    # backward
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    return loss.item(), metric.item()


@torch.no_grad()
def validate_step(model, inps, labs):
    inps = inps.to(device)
    labs = labs.to(device)

    model.eval()  # 设置eval mode

    # forward
    logits = model(inps)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)
    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())  # 返回的是tensor还是标量？

    return loss.item(), metric.item()


def train_model(model, train_dloader, val_dloader, optimizer, scheduler_1r=None, num_epochs=10, print_every=150):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_metric = 0.
    for epoch in range(1, num_epochs+1):
        # 训练
        loss_sum, metric_sum = 0., 0.
        for step, (inps, labs) in enumerate(train_dloader, start=1):
            loss, metric = train_step(model, inps, labs, optimizer)
            loss_sum += loss
            metric_sum += metric

            # 打印batch级别日志
            if step % print_every == 0:
                print('*'*27, f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

        # 验证 一个epoch的train结束，做一次验证
        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inps, labs) in enumerate(val_dloader, start=1):
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
            # checkpoint = save_dir + '{:03d}_{:.3f}_ckpt.tar'.format(epoch, current_metric_avg) ############################################################
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

    plt.savefig(imgs_dir + 'han_'+ metric + '.png')  # 保存图片
    plt.show()


if __name__=='__main__':
    train_dloader = load_data(data_base_dir + 'cnews.train.txt', traindata=True, shuffle=True)
    val_dloader = load_data(data_base_dir + 'cnews.val.txt', traindata=False, shuffle=False)

    print('*' * 27, '%d 个 step:' % len(train_dloader))  # 1000 个step/batch
    sample_batch = next(iter(train_dloader))
    print('*'*27, 'sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[0].dtype,
          sample_batch[1].size(), sample_batch[1].dtype)  # 4   [b, doc_maxlen] int64


    model = MyHAN(sent_maxlen, doc_maxlen, total_words+2, hidden_size, num_classes, embedding_dim)

    torchkeras.summary(model, input_shape=(doc_maxlen, sent_maxlen), input_dtype=torch.int64)

    model = model.to(device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1]

    model.eval()
    sample_out = model(sample_batch[0])
    print('*' * 10, 'sample_out:', sample_out.shape)  # [b, 10]

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = torch.optim.AdamW(params_to_update, lr=LR, weight_decay=1e-4)
    scheduler_1r = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch>EPOCHS*0.6 else 0.5 if epoch>EPOCHS*0.3 else 1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_model(model, train_dloader, val_dloader, optimizer, scheduler_1r,
                num_epochs=EPOCHS, print_every=50)

    plot_metric(df_history, 'loss')
    plot_metric(df_history, metric_name)