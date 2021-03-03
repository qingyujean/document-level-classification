# coding=utf-8
"""
Author  : Jane
Contact : xijian@ict.ac.cn
Time    : 2021/1/29 15:51
Desc:
"""
import torch
import torchkeras

from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import time
import sys

sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from dpcnn.config import *
from dpcnn.prepare_data import load_data


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

        
class DPCNN(torch.nn.Module):
    def __init__(self, num_classes, dropout_p=0.5, net_depth=15, max_seqlen=500):
        super(DPCNN, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = 100
        self.num_filters = 250
        self.depth = net_depth
        self.cur_seq_len = max_seqlen
        self.conv_block_repeat_num = int((net_depth-1)//2) # net_depth- region embedding(1层) -first_conv_block(2层)


        self.embedding = torch.nn.Embedding(num_embeddings=total_words, embedding_dim=embedding_dim,
                                            padding_idx=pad_id) # [b,seq_len]=>[b,seq_len, embedding_dim]
        self.embed_drop1 = torch.nn.Dropout(p=dropout_p)
        self.embed_drop2 = torch.nn.Dropout(p=dropout_p)

        # region embedding层
        # self.resize_embed = torch.nn.Conv2d(in_channels=embedding_dim, out_channels=self.num_filters, kernel_size=(3,self.embedding_dim), stride=1)
        self.resize_embed = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=self.num_filters, kernel_size=3,
                                            stride=1, padding=1) # padding保证等长卷积

        self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # [b, num_filters, seq_len]=>[b, num_filters, seq_len//2]

        self.repeat_layer = torch.nn.ModuleList()

        for i in range(self.conv_block_repeat_num): # 7 层
            if self.cur_seq_len<2: # 长度必须>=2
                break

            _conv_block = torch.nn.Sequential(
                # torch.nn.MaxPool1d(kernel_size=3, stride=2) # [b, num_filters, seq_len]=>[b, num_filters, seq_len//2]

                torch.nn.ReLU(),  # 预激活
                torch.nn.Conv1d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1,
                                padding=1), # [b, num_filters, seq_len//2]=>[b, num_filters, seq_len//2]

                torch.nn.ReLU(),  # 预激活
                torch.nn.Conv1d(in_channels=self.num_filters, out_channels=self.num_filters, kernel_size=3, stride=1,
                                padding=1),  # [b, num_filters, seq_len//2]=>[b, num_filters, seq_len//4]
            )
            self.repeat_layer.append(_conv_block)
            if i>=1:
                self.cur_seq_len = math.ceil(self.cur_seq_len/2.)
            # print('*' * 27, i, 'cur_seq_len:', self.cur_seq_len)

        # self.global_max_pool = torch.nn.MaxPool1d(kernel_size=self.cur_seq_len, stride=self.cur_seq_len)

        # print('*'*27, 'cur_seq_len:', self.cur_seq_len)
        self.fc = torch.nn.Sequential(
            # torch.nn.MaxPool1d(kernel_size=self.cur_seq_len, stride=self.cur_seq_len),
            # glocal max pool: [b, num_filters, cur_seq_len]=>[b, num_filters, 1]
            torch.nn.Flatten(),
            # torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(self.cur_seq_len*self.num_filters, self.num_classes),
            # torch.nn.Linear(self.num_filters, self.num_classes),
        )

    def forward(self, x): # x [b, seq_len]
        x = self.embedding(x) # [b, seq_len]=>[b, seq_len, embedding_dim]
        """
        # x = self.embed_drop(x).unsqueeze(dim=1) # [b, seq_len, embedding_dim]=>[b, 1, seq_len, embedding_dim]
        # x = self.resize_embed(x) # [b, 1, seq_len, embedding_dim]=>[b,250, seq_len-3+1,1] # floor((h-k+2p)/s)+1
        """

        x = self.embed_drop1(x).transpose(1,2) # [b, seq_len, embedding_dim]=>[b, embedding_dim, seq_len]
        x = self.resize_embed(x) # [b, embedding_dim, seq_len]=>[b, num_filters, seq_len] # floor((seq_len-3+2)/1)+1
        x = self.embed_drop2(x)  # [b, num_filters, seq_len]
        # print('*'*27, 'region embedding:', x.shape)

        # x = self.first_conv_block(x) # [b, num_filters, seq_len]=>[b, num_filters, seq_len]

        for i in range(self.conv_block_repeat_num):  # 7 层
            if i==0: # embedding后的第一层
                f_x = self.repeat_layer[i](x) # [b, num_filters, seq_len]=>[b, num_filters, seq_len]
                x = x + f_x # short cut
            else:
                x = self.pool(x) # [b, num_filters, seq_len]=>[b, num_filters, seq_len//2]
                # print('*' * 27, i, x.shape)
                f_x = self.repeat_layer[i](x) # [b, num_filters, seq_len//2]=>[b, num_filters, seq_len//2]
                x = x + f_x  # short cut
            # print('*' * 27, i, x.shape)
        # print('*' * 27, x.shape)
        # x: [b, num_filters, seq_len//(2^6)] 即 [b, num_filters, seq_len//64] 即

        # x = self.global_max_pool(x).squeeze(dim=-1)
        # print('*' * 27, 'after global pool:', x.shape)
        logits = self.fc(x) # [b, num_filters, cur_seq_len]=>[b, num_filters*cur_seq_len]=>[b, num_classes]
        return logits


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

    plt.savefig(imgs_dir + 'dpcnn_'+metric + '.png')  # 保存图片
    plt.show()
        




if __name__=='__main__':
    train_dloader = load_data(data_base_dir + 'cnews.train.txt', traindata=True, shuffle=True)
    val_dloader = load_data(data_base_dir + 'cnews.val.txt', traindata=False, shuffle=False)

    print('*' * 27, '%d 个 step:' % len(train_dloader))  # 1000 个step/batch
    sample_batch = next(iter(train_dloader))
    print('*'*27, 'sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[0].dtype,
          sample_batch[1].size(), sample_batch[1].dtype)  # 4   [b, doc_maxlen] int64

    model = DPCNN(num_classes, net_depth=net_depth, max_seqlen=doc_maxlen)

    torchkeras.summary(model, input_shape=(doc_maxlen,), input_dtype=torch.int64)

    model = model.to(device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1,2,3]

    model.eval()
    sample_out = model(sample_batch[0])
    print('*' * 10, 'sample_out:', sample_out.shape)  # [b, 10]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_1r = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch > EPOCHS * 0.8 else 1)
    train_model(model, train_dloader, val_dloader, optimizer, # scheduler_1r,
                num_epochs=EPOCHS, print_every=50)

    plot_metric(df_history, 'loss')
    plot_metric(df_history, metric_name)