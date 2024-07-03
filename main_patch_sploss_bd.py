import torch
import numpy as np
import warnings
import sklearn.metrics as metrics
import copy
from scipy.io import loadmat
from sklearn import preprocessing
import random
import math
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.nn as nn
import os
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")


class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data = nn.init.constant_(self.bias.data, 0.0)
            # init.xavier_uniform_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_size=256, k=5):
        super(context_embedding, self).__init__()
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size=k)

    def forward(self, x):
        x = self.causal_convolution(x)
        return F.tanh(x)


def load_correlation(path):

    data_dict = loadmat(path)
    data_array = data_dict['connectivity']

    # len_data = data_array.shape[0]

    return data_array


def load_data(path):
    dirs = os.listdir(path)
    # 排序
    dirss = np.sort(dirs)
    print('The dirs have been sorted')
    all = {}
    all_data = []
    for filename in dirss:
        a = np.loadtxt(path + filename)
        a = a.transpose()
        all[filename] = a
        all_data.append(a)
    return all_data


def load_label(path):
    label = []
    with open(path) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            label.append(val)
    label = np.array(label)

    # d_label = np.zeros((len(label), 23))
    # for i in range(len(label)):
    #     d_label[i][label[i]] = 1

    print('load domain label!')
    return label


def get_graph(x, id):
    new_g = np.zeros((1009, 1009))
    for i in range(len(id)):
        for j in range(i+1, len(id)):
            new_g[j][i] = new_g[i][j] = x[id[i], id[j]]
    return new_g


def get_graph_2(g):
    for i in range(len(g)):
        for j in range(i+1, len(g)):
            if g[i][j] > 0.0004:
                g[i][j] = g[j][i] = 1
            else:
                g[i][j] = g[j][i] = 0
    return g


def get_padding_mask(src_len):
    mask = np.zeros((len(src_len), 320))
    # one = np.ones(320)
    for i in range(len(src_len)):
        mask[i, src_len[i]:] = mask[i, src_len[i]:] + 1
    mask_bool = (mask == 1)
    return mask_bool


def data_to_tensor(name, data, mask_data, flag='temporal', device='cuda'):
    data = np.array(data)  # -1 200 90
    pro = preprocessing.MinMaxScaler()
    # data_pos = np.zeros((data.shape[0], data.shape[1], data.shape[2]))  # -1 90 200
    if flag == 'temporal':
        data_pos = np.zeros((data.shape[0], data.shape[1], data.shape[2]))  # -1 90 200
        pos = np.load('data/preprocess/'+name+'/position_temporal_ml500.npy')  # 1 200 500
        for i in range(data.shape[0]):
            data_ln = int(np.sum(1-mask_data[i]))
            data_pos[i, :, :data_ln] = data[i, :, :data_ln] + pos[:, :, :data_ln]
            data_pos[i] = pro.fit_transform(data_pos[i])
    else:
        data_pos = data  # -1 90 200
        # for k in range(data.shape[0]):
        #     for i in range(data.shape[1]):
        #         for j in range(i + 1, data.shape[2]):
        #             if data[k][i][j] < 0.5 or data[k][i][j] < -0.5:
        #                 data_pos[k][i][j] = data_pos[k][j][i] = 0
    data_pos = torch.FloatTensor(data_pos).to(device)
    print('to tensor finished!')
    return data_pos


def get_mask_label(label):
    l = len(label)
    m = torch.zeros((l, l)).to(device)
    for i in range(l):
        for j in range(i, l):
            if label[i] == label[j]:
                m[i][j] = m[j][i] = 1
    return m


# 定义clone函数，因为在transformer中很多重复的模块，clone方便
# 其实就是把for循环拎出来写，用copy复制多层（copy可以保证复制的module训练时参数不同）
def clone(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query, key, value, mask=None, dropout=None):
    '''
    注意力机制实现
    query: 查询向量，理解：原文本
    key: 理解：给的参考
    value: 理解：学习出的内容
    mask: 掩码
    '''

    # 获得词嵌入的维度
    d_size = query.size(-1)
    # 按照公式计算注意力张量
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_size)

    # 是否使用掩码张量
    if mask is not None:
        # 用-1e9替换score中的值，替换位置是mask为0的地方
        scores = scores.masked_fill(mask, -1e9)

    p_attn = F.softmax(scores, dim=-2)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 返回：attention得分，注意力张量

    return torch.matmul(p_attn, value), p_attn


def patch_attention(query, key, value, win_size, stride, mask=None, dropout=None):
    '''
    注意力机制实现
    query: 查询向量，理解：原文本
    key: 理解：给的参考
    value: 理解：学习出的内容
    mask: 掩码
    '''

    def cal_score(query, key):
        d_size = query.size(-1)
        batch = query.size(0)
        l = query.shape[-2]
        n_l = int((l-2) / win_size)
        tmp = torch.zeros(n_l, l).to('cuda')
        score = torch.zeros(l, l).to('cuda')
        for i in range(n_l):
            if i == 0:
                tmp[i, i] = 1
            else:
                tmp[i, win_size * i] = 1
        for wi in range(win_size):
            q = query[:, :, :l-2, :]
            q = q.reshape(batch, -1, n_l, win_size, d_size)

            for wj in range(win_size):
                k = key[:, :, :l-2, :]
                k = k.reshape(batch, -1, n_l, win_size, d_size)
                s = torch.einsum("bhmld,bhnld->bhmn", q, k)
                score = score + torch.matmul(tmp.t(), torch.matmul(s, tmp))
                key = torch.roll(key, shifts=l - 1, dims=-2)
                score = torch.roll(score, shifts=l - 1, dims=-1)
            key = torch.roll(key, shifts=win_size, dims=-2)
            score = torch.roll(score, shifts=win_size, dims=-1)
            query = torch.roll(query, shifts=l - 1, dims=-2)
            score = torch.roll(score, shifts=l - 1, dims=-2)
        query = torch.roll(query, shifts=win_size, dims=-2)
        score = torch.roll(score, shifts=win_size, dims=-2)
        return score/(math.sqrt(d_size*win_size))

    scores = cal_score(query, key)

    # 是否使用掩码张量
    if mask is not None:
        # 用-1e9替换score中的值，替换位置是mask为0的地方
        scores = scores.masked_fill(mask, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 返回：attention得分，注意力张量

    return torch.matmul(p_attn, value), p_attn


def patch_attention2(query, key, value, win_size, stride, mask=None, dropout=None):
    '''
    注意力机制实现
    query: 查询向量，理解：原文本
    key: 理解：给的参考
    value: 理解：学习出的内容
    mask: 掩码
    '''

    def cal_score(query, key):
        d_size = query.size(-1)
        batch = query.size(0)
        l = query.shape[-2]
        n_l = int((l-2) / win_size)
        tmp = torch.zeros(n_l, l).to('cuda')
        score = torch.zeros(l, l).to('cuda')
        for i in range(n_l):
            if i == 0:
                tmp[i, i] = 1
            else:
                tmp[i, win_size * i] = 1
        for wi in range(win_size):
            q = query[:, :, :l-2, :]
            q = q.reshape(batch, -1, n_l, win_size, d_size)

            for wj in range(win_size):
                k = key[:, :, :l-2, :]
                k = k.reshape(batch, -1, n_l, win_size, d_size)
                s = torch.einsum("bhmld,bhnld->bhmn", q, k)
                score = score + torch.matmul(tmp.t(), torch.matmul(s, tmp))
                key = torch.roll(key, shifts=l - 1, dims=-2)
                score = torch.roll(score, shifts=l - 1, dims=-1)
            key = torch.roll(key, shifts=win_size, dims=-2)
            score = torch.roll(score, shifts=win_size, dims=-1)
            query = torch.roll(query, shifts=l - 1, dims=-2)
            score = torch.roll(score, shifts=l - 1, dims=-2)
        query = torch.roll(query, shifts=win_size, dims=-2)
        score = torch.roll(score, shifts=win_size, dims=-2)
        return score/(math.sqrt(d_size*win_size))

    scores = cal_score(query, key)

    # 是否使用掩码张量
    if mask is not None:
        # 用-1e9替换score中的值，替换位置是mask为0的地方
        scores = scores.masked_fill(mask, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 返回：attention得分，注意力张量

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention2(nn.Module):
    def __init__(self, embedding_dim, head, dropout=0.1):
        '''
        head: 头数
        embedding_dim: 词嵌入维度
        causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel
        '''

        super(MultiHeadAttention2, self).__init__()

        # 使用测试中常见的assert语句判断head是否能被embedding_dim整除
        hidden_size = 64
        assert hidden_size % head == 0

        self.d_k = hidden_size // head
        self.head = head

        # 初始化4个liner层，三个用于q，k，v变换，一个用于拼接矩阵后的变换
        self.linears1 = clone(context_embedding(embedding_dim, hidden_size, 3), 2)
        self.linears2 = clone(nn.Linear(embedding_dim, hidden_size, bias=False), 3)
        self.linear2 = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):  # q k 32 200 320  v 32 320 200
        if mask is not None:
            # 扩展mask维度，对应multihead
            mask = mask.unsqueeze(1).unsqueeze(-1)

        batch_size = query.size(0)
        # query1, key1 = [model(x) for model, x in zip(self.linears1, (query, key))]   # 32 64 320
        # value1 = self.linear2(value)
        query2, key2, value2 = [model(x) for model, x in zip(self.linears2, (query, key, value))]
        # 32 320 64

        # # point
        query2 = query2.view(batch_size, 8, -1, self.head, self.d_k)
        key2 = key2.view(batch_size, 8, -1, self.head, self.d_k)
        value2 = value2.view(batch_size, 8, -1, self.head, self.d_k)

        # # patch
        query = query2.reshape(batch_size*10, 8, -1, self.head, self.d_k).transpose(3, 2)
        key = key2.reshape(batch_size*10, 8, -1, self.head, self.d_k).transpose(3, 2)
        value = value2.reshape(batch_size*10, 8, -1, self.head, self.d_k).transpose(3, 2)

        # temporal
        # 计算attention
        x, attn = attention(query, key, value, mask, self.dropout)

        # x是多头的注意力结果，需要将头拼接
        # 维度转换：batch_size*length*head*切片后的维度
        x = x.transpose(2, 3).contiguous().view(batch_size, 8, -1, self.head*self.d_k)
        x = torch.sum(x, 1)

        # 拼接后的结果进入liner层
        return self.linear(x), attn.sum(2)


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward=1024, dropout=0.1):
        super(FFN, self).__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x, x3):
        x = x + self.dropout1(x3)
        tmp = x = self.norm1(x)

        x4 = self.linear2(self.dropout(self.activation(self.linear1(x))))

        x = x + self.dropout2(x4)
        x = self.norm2(x)

        return x, tmp


class TModule(nn.Module):
    def __init__(self, d_model, head, dim_feedforward=1024, dropout=0.1):
        super(TModule, self).__init__()
        self.dim = d_model

        self.self_attn = MultiHeadAttention2(d_model, head, dropout=dropout)  # 90 30 200 token
        self.FNN = FFN(d_model, dim_feedforward, dropout)

        # self.self_attn2 = MultiHeadAttention2(d_model, head, dropout=dropout)  # 90 30 200 token
        # self.FNN2 = FFN(d_model, dim_feedforward, dropout)
        #
        # self.self_attn3 = MultiHeadAttention2(d_model, head, dropout=dropout)  # 90 30 200 token
        # self.FNN3 = FFN(d_model, dim_feedforward, dropout)
        #
        # self.self_attn4 = MultiHeadAttention2(d_model, head, dropout=dropout)  # 90 30 200 token
        # self.FNN4 = FFN(d_model, dim_feedforward, dropout)
        #
        # self.self_attn5 = MultiHeadAttention2(d_model, head, dropout=dropout)  # 90 30 200 token
        # self.FNN5 = FFN(d_model, dim_feedforward, dropout)
        #
        # self.self_attn6 = MultiHeadAttention2(d_model, head, dropout=dropout)  # 90 30 200 token
        # self.FNN6 = FFN(d_model, dim_feedforward, dropout)
        #
        # self.self_attn7 = MultiHeadAttention2(d_model, head, dropout=dropout)  # 90 30 200 token
        # self.FNN7 = FFN(d_model, dim_feedforward, dropout)

        self.fc = nn.Linear(d_model, 2)
        self.sig = nn.Sigmoid()

        self.weight = nn.Parameter(torch.FloatTensor(d_model, 8))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, src):

        w = F.relu(self.weight)

        tmp = torch.eye(8, 8).unsqueeze(-1).unsqueeze(-1).to(device)
        dmf = torch.multiply(
            torch.matmul(w.unsqueeze(1).permute(2, 0, 1), w.unsqueeze(0).permute(2, 0, 1)),
            tmp
        )  # 8 8 200 200
        dmfs = torch.sum(dmf, 0)  # 8 200 200
        # FF = torch.sum(dmfs, 0)  # 200 200
        srcf = torch.matmul(torch.repeat_interleave(src.unsqueeze(1), 8, 1),
                            torch.repeat_interleave(dmfs.unsqueeze(0), src.shape[0], 0))  # 32 8 950 116
        # sq = srcf.reshape(-1, 950, self.dim).permute(0, 2, 1)

        # temporal
        # src3, attn = self.self_attn(src.permute(0, 2, 1), src.permute(0, 2, 1), srcf)  # 32 320 200
        src3, attn = self.self_attn(srcf, srcf, srcf)  # 32 320 200
        out0, _ = self.FNN(src, src3)  # 32 320 200

        out0 = (torch.sum(out0, dim=1) / 950).squeeze()
        out = self.fc(out0)
        out = self.sig(out)

        ta = torch.sum(attn, -1)  # 32 320
        # dmfa = torch.multiply(dmfs.unsqueeze(0).unsqueeze(-1), ta.unsqueeze(-2).unsqueeze(-2))
        dmfs = dmfs.unsqueeze(0).unsqueeze(-1)
        ta = ta.unsqueeze(-2).unsqueeze(-2)

        dmfas = (torch.multiply(dmfs[:, 0, :], ta[:, 0, :]) +
                 torch.multiply(dmfs[:, 1, :], ta[:, 1, :]) +
                 torch.multiply(dmfs[:, 2, :], ta[:, 2, :]) +
                 torch.multiply(dmfs[:, 3, :], ta[:, 3, :]) +
                 torch.multiply(dmfs[:, 4, :], ta[:, 4, :]) +
                 torch.multiply(dmfs[:, 5, :], ta[:, 5, :]) +
                 torch.multiply(dmfs[:, 6, :], ta[:, 6, :]) +
                 torch.multiply(dmfs[:, 7, :], ta[:, 7, :])
                 )
        # # dmfas3 = torch.multiply(dmfs[:, 3, :], ta[:, 3, :])
        # dmfas = dmfas + torch.multiply(dmfs[:, 4:6, :], ta[:, 4:6, :]).sum(1)
        # # dmfas5 = torch.multiply(dmfs[:, 5, :], ta[:, 5, :])
        # dmfas = dmfas + torch.multiply(dmfs[:, 6:8, :], ta[:, 6:8, :]).sum(1)
        # # dmfas7 = torch.multiply(dmfs[:, 7, :], ta[:, 7, :])
        dmfas = dmfas.view(int(dmfas.shape[0]/10), 10, self.dim, self.dim, 95).sum(1)

        ssrc = src.view(src.shape[0], 95, 10, 116)  # 32  95 10 116
        ssrc = torch.matmul(ssrc.permute(0, 1, 3, 2), ssrc).permute(0, 2, 3, 1)
        # ssrc = torch.repeat_interleave(ssrc, dmfas.shape[0], 0)
        # ssrc = ssrc.permute(0, 2, 3, 1)
        FF = torch.matmul(w, w.T)
        diag_elements = FF.diag()
        # ortho_loss_matrix = torch.square(FF - torch.diag(diag_elements))
        ortho_loss = torch.sum(diag_elements)
        variance_loss = diag_elements.var()
        neg_loss = torch.sum(F.relu(torch.tensor(1e-6) - self.weight))

        loss = 0.1 * torch.norm(ssrc - dmfas) + 10 * torch.norm(w, p=1) + neg_loss + ortho_loss + variance_loss

        return out, loss


class Mymodule(nn.Module):
    def __init__(self, d_model, head, dim_feedforward=1024, dropout=0.1):
        super(Mymodule, self).__init__()
        self.device = device
        self.head = head
        self.t_module = TModule(d_model, self.head, dim_feedforward, dropout)
        # self.t_module2 = TModule(d_model, self.head, dim_feedforward, dropout)

    def forward(self, src):  # 32 200 90
        output, loss = self.t_module(src)  # 90 32 200

        return output, loss

#
# def evaluate(data, sdata, label, mask,
#              # g, dg,
#              dl,
#              # ng,
#              model, flag, device='cpu'):
#     model.eval()
#     preds = []
#
#     with torch.no_grad():
#         output, fl = model(data, sdata, mask)
#         _, indices = torch.max(output, 1)
#         preds.append(indices.cpu().data.numpy())
#
#     labels = label.cpu().data.numpy()
#     preds = np.hstack(preds)
#
#     if flag == 'train':
#         loss = F.cross_entropy(output, label)  # + F.cross_entropy(fl, dl)
#         result = {'prec': metrics.precision_score(labels, preds),
#                   'recall': metrics.recall_score(labels, preds),
#                   'acc': metrics.accuracy_score(labels, preds),
#                   'F1': metrics.f1_score(labels, preds),
#                   'auc': metrics.roc_auc_score(labels, preds),
#                   'matrix': confusion_matrix(labels, preds)}
#     else:
#         loss = F.cross_entropy(output[len(data)-len(labels):], label)  # + F.cross_entropy(fl[len(data)-len(labels):], dl)
#         result = {'prec': metrics.precision_score(labels, preds[len(data)-len(labels):]),
#                   'recall': metrics.recall_score(labels, preds[len(data)-len(labels):]),
#                   'acc': metrics.accuracy_score(labels, preds[len(data)-len(labels):]),
#                   'F1': metrics.f1_score(labels, preds[len(data)-len(labels):]),
#                   'auc': metrics.roc_auc_score(labels, preds[len(data)-len(labels):]),
#                   'matrix': confusion_matrix(labels, preds[len(data)-len(labels):])}
#
#     return loss, result


def evaluate(data, label,
             model, flag, device='cpu'):
    model.eval()
    preds = []

    with torch.no_grad():
        output, fl = model(data)
        _, indices = torch.max(output, 1)
        preds.append(indices.cpu().data.numpy())

    labels = label.cpu().data.numpy()
    preds = np.hstack(preds)

    if flag == 'train':
        loss = F.cross_entropy(output, label)  # + F.cross_entropy(fl, dl)
        result = {'prec': metrics.precision_score(labels, preds),
                  'recall': metrics.recall_score(labels, preds),
                  'acc': metrics.accuracy_score(labels, preds),
                  'F1': metrics.f1_score(labels, preds),
                  'auc': metrics.roc_auc_score(labels, preds),
                  'matrix': confusion_matrix(labels, preds)}
    else:
        loss = F.cross_entropy(output, label)  # + F.cross_entropy(fl[len(data)-len(labels):], dl)
        p = metrics.precision_score(labels, preds[len(data) - len(labels):])
        r = metrics.recall_score(labels, preds[len(data) - len(labels):])
        ac = metrics.accuracy_score(labels, preds[len(data) - len(labels):])
        f1 = metrics.f1_score(labels, preds[len(data) - len(labels):])
        au = metrics.roc_auc_score(labels, preds[len(data) - len(labels):])
        result = [p, r, ac, f1, au]

    return loss, result


def train(brain_node, train_data, train_label,
          test_data=None, test_label=None,
          device='cuda'):

    batch = 32
    batch_epoch = math.ceil(len(train_data)/batch)

    input_size = brain_node
    head = 8
    model = Mymodule(input_size, head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    max_epoch = 80
    rr = []

    for epoch in range(max_epoch):
        model.train()
        avg_loss = 0
        for b in range(batch_epoch):
            model.zero_grad()
            if b == batch_epoch-1:
                if train_label[b * batch:].shape[0] == 1:
                    continue
                output, at = model(train_data[b * batch:],
                                   )
                loss = F.cross_entropy(output, train_label[b * batch:], weight=torch.tensor([0.7, 1]).to(device)) + 0.00001*at
                # loss = F.cross_entropy(output, train_label[b * batch:]) + 0.00001*at

            else:
                output, at = model(train_data[b * batch:(b + 1) * batch],
                                   )
                loss = F.cross_entropy(output, train_label[b * batch:(b + 1) * batch], weight=torch.tensor([0.7, 1]).to(device)) + 0.00001*at
                # loss = F.cross_entropy(output, train_label[b * batch:(b + 1) * batch]) + 0.00001*at

            loss.backward()
            avg_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            # CosineLR.step()
            optimizer.zero_grad()
            # CosineLR.zero_grad()

        avg_loss /= batch_epoch

        # if epoch % 5 == 0 or epoch == max_epoch-1:
        if epoch > max_epoch - 20:
            print('epoch:::', epoch, '----loss:::', avg_loss)
        #     train_loss, train_result = evaluate(train_data[:100, :],
        #                                         train_label[:100],
        #                                         model,
        #                                         'train',
        #                                         device)
        #     print('train:', train_result)

            if test_data is not None:
                test_loss, test_result = evaluate(test_data,
                                                  test_label,
                                                  model,
                                                  'text',
                                                  device)
                rr.append(test_result)
                print('test:', test_result)
                print('--------------test loss------------', test_loss)
            if epoch > 65:
                torch.save(model.state_dict(), 'draw/interpretation/bd_stage/epoch' + str(epoch) + '_cv' + str(i) + '_checkpoint.pt')
    return rr, model


if __name__ == '__main__':
    # 导入数据
    brain_node = 116
    num = 10
    results = []
    device = 'cuda'

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # dgl.seed(seed)

    setup_seed(0)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data = np.load('data/MDD/BP_HC_sig.npy')  # 427 950 116
    label = np.load('data/MDD/BP_HC_label.npy')  # 427

    zip_list = list(zip(data, label))
    random.Random(0).shuffle(zip_list)
    data, label = zip(*zip_list)

    data = np.array(data)
    label = np.array(label)

    kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(data, label)):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = label[train_index], label[test_index]
        # # setup_seed(0)
        # if i != 7:
        #     continue

        train_data = torch.FloatTensor(train_data).to(device)
        test_data = torch.FloatTensor(test_data).to(device)

        train_label = torch.LongTensor(train_label).to(device)
        test_label = torch.LongTensor(test_label).to(device)

        print('***************load data finished!*********************', i)

        result, model = train(brain_node, train_data, train_label,
                              test_data, test_label,
                              device=device)
        results.append(result)
        # torch.save(model.state_dict(), 'checkpoint/' + str(i) + '_checkpoint.pt')

    res = np.array(results)
    res = np.mean(res, axis=0)
    print('--------------------10cv results----------------------')
    print(res)





