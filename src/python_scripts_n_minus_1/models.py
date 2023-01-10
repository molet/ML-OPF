from math import floor

import numpy as np
import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    ChebConv,
    SplineConv,
    GATConv,
    GraphConv,
    BatchNorm,
)


def cnn_output_dim(input_size, n_conv_layers, k=3, p=0, s=1, mp=2):
    for i in range(n_conv_layers):
        output_size = (1 + (input_size - k + 2 * p) / s) / (mp if mp > 0 else 1)
        input_size = output_size
    return floor(output_size)


class FCNN(torch.nn.Module):
    def __init__(self, parameters):
        super(FCNN, self).__init__()
        self.params = parameters
        for key in self.params:
            setattr(self, key, self.params[key])
        self.chain = nn.ModuleList()
        for l in list(self.layers.keys())[:-1]:
            self.chain.append(
                nn.Sequential(
                    nn.Linear(self.layers[l][0], self.layers[l][1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.layers[l][1]),
                )
            )
        self.out_layer = nn.Sequential(
            nn.Linear(
                self.layers[list(self.layers.keys())[-1]][0],
                self.layers[list(self.layers.keys())[-1]][1],
            )
        )

    def forward(self, x):
        for layer in self.chain:
            x = layer(x)
        x = self.out_layer(x)
        if self.primal:
            x = torch.sigmoid(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self, parameters):
        super(CNN, self).__init__()
        self.params = parameters
        for key in self.params:
            setattr(self, key, self.params[key])
        self.chain = nn.ModuleList()
        self.readout_input_dim = cnn_output_dim(
            self.num_bus, len(list(self.layers.keys())[:-1])
        )
        self.readout_hidden_dim = int(
            round(
                np.mean(
                    [
                        self.layers[list(self.layers.keys())[-2]][1]
                        * self.readout_input_dim,
                        self.layers[list(self.layers.keys())[-1]][0],
                    ]
                )
            )
        )
        for l in list(self.layers.keys())[:-1]:
            self.chain.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.layers[l][0],
                        self.layers[l][1],
                        kernel_size=3,
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.BatchNorm1d(self.layers[l][1]),
                )
            )
        self.fc = nn.Sequential(
            nn.Linear(
                self.layers[list(self.layers.keys())[-2]][1] * self.readout_input_dim,
                self.layers[list(self.layers.keys())[-1]][0],
            ),
        )

    def forward(self, x):
        for layer in self.chain:
            x = layer(x)
        x = x.view(x.size()[0], x.size()[1] * x.size()[2])
        x = self.fc(x)
        if self.primal:
            x = torch.sigmoid(x)
        return x


class GNN(torch.nn.Module):
    def batch_norm(self, bn, X):
        return [x.reshape(X[0].shape) for x in bn(torch.stack(X).reshape((len(X), -1)))]

    def __init__(self, parameters):
        super(GNN, self).__init__()
        self.params = parameters
        for key in self.params:
            setattr(self, key, self.params[key])
        self.chain_conv = nn.ModuleList()
        self.chain_conv_bn = nn.ModuleList()
        self.chain = nn.ModuleList()
        self.chain_bn = nn.ModuleList()
        for l in list(self.gnn_layers.keys()):
            if self.architecture == "gcn":
                self.chain_conv.append(
                    GCNConv(
                        self.gnn_layers[l][0],
                        self.gnn_layers[l][1],
                        cached=True,
                    )
                )
            elif self.architecture == "chnn":
                self.chain_conv.append(
                    ChebConv(
                        self.gnn_layers[l][0],
                        self.gnn_layers[l][1],
                        K=self.kernel_size,
                    )
                )
            elif self.architecture == "snn":
                self.chain_conv.append(
                    SplineConv(
                        self.gnn_layers[l][0],
                        self.gnn_layers[l][1],
                        dim=1,
                        kernel_size=self.kernel_size,
                    )
                )
            elif self.architecture == "gat":
                self.chain_conv.append(
                    GATConv(
                        self.gnn_layers[l][0],
                        self.gnn_layers[l][1],
                        heads=1,
                    )
                )
            elif self.architecture == "gc":
                self.chain_conv.append(
                    GraphConv(
                        self.gnn_layers[l][0],
                        self.gnn_layers[l][1],
                        aggr="add",
                    )
                )
            self.chain_conv_bn.append(
                torch_geometric.nn.BatchNorm(self.gnn_layers[l][1])
            )
        if self.primal & self.local:
            return
        for l in list(self.fcnn_layers.keys())[:-1]:
            self.chain.append(nn.Linear(self.fcnn_layers[l][0], self.fcnn_layers[l][1]))
            self.chain_bn.append(nn.BatchNorm1d(self.fcnn_layers[l][1]))
        self.out_layer = nn.Sequential(
            nn.Linear(
                self.fcnn_layers[list(self.fcnn_layers.keys())[-1]][0],
                self.fcnn_layers[list(self.fcnn_layers.keys())[-1]][1],
            )
        )

    def forward(self, batch):
        X, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        for layer, bn in zip(self.chain_conv, self.chain_conv_bn):
            if self.architecture == "gat":
                X = layer(X, edge_index)
            else:
                X = layer(X, edge_index, edge_attr)
            X = F.relu(X)
            X = bn(X)
        if self.primal & self.local:
            return torch.sigmoid(X)
        X = X.reshape(self.batch_size, -1)
        for layer, bn in zip(self.chain, self.chain_bn):
            X = [layer(x) for x in X]
            X = [F.relu(x) for x in X]
            X = self.batch_norm(bn, X)
        X = [self.out_layer(x) for x in X]
        if self.primal:
            X = [torch.sigmoid(x) for x in X]
        return X
