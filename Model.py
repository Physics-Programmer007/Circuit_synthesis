import numpy as np
import igraph
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

# Some utility functions

NODE_TYPE = {
    'R': 0,
    'C': 1,
    '+gm+': 2,
    '-gm+': 3,
    '+gm-': 4,
    '-gm-': 5,
    'sudo_in': 6,
    'sudo_out': 7,
    'In': 8,
    'Out': 9
}

SUBG_NODE = {
    0: ['In'],
    1: ['Out'],
    2: ['R'],
    3: ['C'],
    4: ['R', 'C'],
    5: ['R', 'C'],
    6: ['+gm+'],
    7: ['-gm+'],
    8: ['+gm-'],
    9: ['-gm-'],
    10: ['C', '+gm+'],
    11: ['C', '-gm+'],
    12: ['C', '+gm-'],
    13: ['C', '-gm-'],
    14: ['R', '+gm+'],
    15: ['R', '-gm+'],
    16: ['R', '+gm-'],
    17: ['R', '-gm-'],
    18: ['C', 'R', '+gm+'],
    19: ['C', 'R', '-gm+'],
    20: ['C', 'R', '+gm-'],
    21: ['C', 'R', '-gm-'],
    22: ['C', 'R', '+gm+'],
    23: ['C', 'R', '-gm+'],
    24: ['C', 'R', '+gm-'],
    25: ['C', 'R', '-gm-']
}

SUBG_CON = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: 'series',
    5: 'parral',
    6: None,
    7: None,
    8: None,
    9: None,
    10: 'parral',
    11: 'parral',
    12: 'parral',
    13: 'parral',
    14: 'parral',
    15: 'parral',
    16: 'parral',
    17: 'parral',
    18: 'parral',
    19: 'parral',
    20: 'parral',
    21: 'parral',
    22: 'series',
    23: 'series',
    24: 'series',
    25: 'series'
}

SUBG_INDI = {0: [],
             1: [],
             2: [0],
             3: [1],
             4: [0, 1],
             5: [0, 1],
             6: [2],
             7: [2],
             8: [2],
             9: [2],
             10: [1, 2],
             11: [1, 2],
             12: [1, 2],
             13: [1, 2],
             14: [0, 2],
             15: [0, 2],
             16: [0, 2],
             17: [0, 2],
             18: [1, 0, 2],
             19: [1, 0, 2],
             20: [1, 0, 2],
             21: [1, 0, 2],
             22: [1, 0, 2],
             23: [1, 0, 2],
             24: [1, 0, 2],
             25: [1, 0, 2]
             }


def one_hot(idx, length):
    if type(idx) in [list, range]:
        if idx == []:
            return None
        idx = torch.LongTensor(idx).unsqueeze(0).t()
        x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    else:
        idx = torch.LongTensor([idx]).unsqueeze(0)
        x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


def inverse_adj(adj, device):
    n_node = adj.size(0)
    aug_adj = adj + torch.diag(torch.ones(n_node).to(device))
    aug_diag = torch.sum(aug_adj, dim=0)
    return torch.diag(1 / aug_diag)


class subc_GNN(nn.Module):
    def __init__(self, num_cat, out_feat, dropout=0.5, num_layer=2, readout='sum', device=None):
        super(subc_GNN, self).__init__()
        self.catag_lin = nn.Linear(num_cat, out_feat)
        self.numer_lin = nn.Linear(1, out_feat)
        self.layers = nn.ModuleList()
        self.emb_dim = 2 * out_feat
        self.dropout = dropout
        self.num_cat = num_cat
        # linlayer = nn.Linear(self.emb_dim, self.emb_dim)
        # act = nn.ReLu()
        # self.layers.append(linlayer)
        # self.layers.append(act)
        # if self.dropout > 0.0001:
        #    drop = nn.Dropout(dropout)
        #    self.layers.append(drop)
        for i in range(num_layer):
            linlayer = nn.Linear(self.emb_dim, self.emb_dim)
            act = nn.ReLU()
            self.layers.append(linlayer)
            if self.dropout > 0.0001:
                drop = nn.Dropout(dropout)
                self.layers.append(drop)
            self.layers.append(act)
        self.device = device
        self.readout = readout
        self.num_layer = num_layer

    def forward(self, G):
        # G is a batch of graphs
        nodes_list = [g.vcount() for g in G]
        num_graphs = len(nodes_list)
        num_nodes = sum(nodes_list)
        sub_nodes_types = []
        sub_nodes_feats = []
        num_subg_nodes = []
        for i in range(num_graphs):
            g = G[i]
            for j in range(nodes_list[i]):
                sub_nodes_types += g.vs[j]['subg_ntypes']
                sub_nodes_feats += g.vs[j]['subg_nfeats']
                num_subg_nodes.append(len(g.vs[j]['subg_ntypes']))
        all_nodes = sum(num_subg_nodes)
        all_adj = torch.zeros(all_nodes, all_nodes)
        node_count = 0
        for i in range(num_graphs):
            g = G[i]
            for j in range(nodes_list[i]):
                adj_flat = g.vs[j]['subg_adj']
                subg_n = len(g.vs[j]['subg_ntypes'])
                all_adj[node_count:node_count + subg_n, node_count:node_count + subg_n] = torch.FloatTensor(
                    adj_flat).reshape(subg_n, subg_n)
                node_count += subg_n
        all_adj = all_adj.to(self.get_device())
        in_categ = self._one_hot(sub_nodes_types, self.num_cat)
        in_numer = torch.FloatTensor(sub_nodes_feats).to(self.get_device()).unsqueeze(0).t()
        # print(in_categ)
        # print(in_numer)
        # print(all_adj)
        in_categ = self.catag_lin(in_categ)
        in_numer = self.numer_lin(in_numer)
        x = torch.cat([in_categ, in_numer], dim=1)
        inv_deg = inverse_adj(all_adj)
        # print(in_categ)
        # print(in_numer)
        # print(inv_deg)
        if self.dropout > 0.0001:
            for i in range(self.num_layer - 1):
                x = self.layers[3 * i](x)
                x = x + torch.matmul(all_adj, x)
                x = torch.matmul(inv_deg, x)
                x = self.layers[3 * i + 1](x)
                x = self.layers[3 * i + 2](x)
            x = self.layers[3 * (i + 1)](x)
            x = x + torch.matmul(all_adj, x)
            x = torch.matmul(inv_deg, x)
            x = self.layers[3 * (i + 1) + 2](x)
        else:
            for i in range(self.num_layer - 1):
                x = self.layers[2 * i](x)
                x = x + torch.matmul(all_adj, x)
                x = torch.matmul(inv_deg, x)
                x = self.layers[2 * i + 1](x)
                # x = self.layers[3 * i + 2](x)
            x = self.layers[2 * (i + 1)](x)
            x = x + torch.matmul(all_adj, x)
            x = torch.matmul(inv_deg, x)
            x = self.layers[2 * (i + 1) + 1](x)
        # readout phase
        # out = torch.zeros(num_nodes, self.emb_dim).to(self.get_device())
        out = x
        node_count = 0
        new_G = []
        for i in range(num_graphs):
            g = G[i].copy()
            for j in range(nodes_list[i]):
                subg_n = len(g.vs[j]['subg_ntypes'])
                subg_represent = out[node_count:node_count + subg_n, :]
                if self.readout == 'sum':
                    subg_feat = torch.sum(subg_represent, dim=0)
                elif self.readout == 'mean':
                    subg_feat = torch.mean(subg_represent, dim=0)
                else:
                    subg_feat = None
                    raise MyException('Undefined pool method')
                g.vs[j]['subg_feat'] = subg_feat
                node_count += subg_n
            new_G.append(g)
        return new_G

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

