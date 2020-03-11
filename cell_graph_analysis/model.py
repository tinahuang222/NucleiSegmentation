""" Graph convolutional model for cell graph cancer detection

Author: Samir Akre
"""
import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch_geometric.nn import BatchNorm, SAGPooling, SAGEConv
from torch.nn import Linear


class Net(torch.nn.Module):
    def __init__(self, in_feats):
        super(Net, self).__init__()

        hs_1 = in_feats * 2
        self.conv1 = SAGEConv(in_feats, hs_1)
        self.bn1 = BatchNorm(hs_1)
        self.pool1 = SAGPooling(hs_1, ratio=0.5)

        hs_2 = int(hs_1 * 2)
        self.conv2 = SAGEConv(hs_1, hs_2)
        self.bn2 = BatchNorm(hs_2)
        self.pool2 = SAGPooling(hs_2, ratio=0.5)

        num_classes = 2
        self.lin1 = Linear(hs_2, num_classes).cuda()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool1(
          x, edge_index, batch=data.batch
        )

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x, edge_index, edge_attr, batch, perm, score = self.pool2(
          x, edge_index, batch=batch
        )

        x = nn.global_mean_pool(x, batch)
        x = F.relu(x)
        x = self.lin1(x)

        return F.softmax(x, dim=1)
