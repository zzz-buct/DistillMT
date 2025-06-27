import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class HGT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_heads=4, num_layers=2, dropout=0.5, with_bn=True, with_bias=True):
        super(HGT, self).__init__()

        self.num_layers = num_layers
        self.with_bn = with_bn
        self.hgt_layers = nn.ModuleList()
        self.bns = nn.ModuleList() if with_bn else None
        self.dropout = dropout
        self.name = "HGT"
        self.nhid = nhid
        self.num_heads = num_heads

        for i in range(num_layers):
            in_channels = nfeat if i == 0 else nhid
            out_channels = nhid
            heads = num_heads if i < num_layers - 1 else 1

            self.hgt_layers.append(
                HGTConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    metadata=(['node'], [('node', 'edge', 'node')]),
                    heads=heads
                )
            )

            if with_bn and i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(self, x_dict, edge_index_dict):
        for i, layer in enumerate(self.hgt_layers):
            x_dict = layer(x_dict, edge_index_dict)

            if i != len(self.hgt_layers) - 1:
                # print(f"[HGT] Layer {i} output shape: {x_dict['node'].shape}")  # ✅ 打印一下
                if self.with_bn:
                    x_dict['node'] = self.bns[i](x_dict['node'])
                x_dict['node'] = F.relu(x_dict['node'])
                x_dict['node'] = F.dropout(x_dict['node'], p=self.dropout, training=self.training)

        return x_dict['node']

    def initialize(self):
        for m in self.hgt_layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()
