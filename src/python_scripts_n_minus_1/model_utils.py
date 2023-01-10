from math import floor
import json
import time

import numpy as np
import torch
import torch.nn as nn

import models
from log import getLogger

logger = getLogger(__name__)


def construct_model_parameters(
    train_set, architecture, architecture_size, batch_size, primal, kernel_size, local, num_bus_prime, num_output
):

    model_parameters = {
        "architecture": architecture,
        "architecture_size": architecture_size,
        "batch_size": batch_size,
        "primal": primal,
        "kernel_size": kernel_size,
        "local": local,
    }

    if architecture in ["fcnn", "fcnn1"]:
        input_size = train_set.dataset.X[0].size(0)
        model_parameters["num_output_cols"] = 2
        output_size = (
            num_output
            if primal
            else train_set.dataset.Y[0].size(0)
        )
        num_hidden_layers = 1 if architecture == "fcnn1" else 3
        model_parameters["layers"] = fcnn_layers(
            input_size, output_size, num_hidden_layers
        )

    elif architecture == "cnn":
        model_parameters["num_bus"] = len(train_set.dataset.X[0][0])
        input_size = train_set.dataset.X[0].size(1)
        model_parameters["num_output_cols"] = 2
        output_size = (
            num_output
            if primal
            else train_set.dataset.Y[0].size(0)
        )
        model_parameters["layers"] = {
            "0": [len(train_set.dataset.X[0]), 4],
            "1": [4, 8],
            "2": [8, 16],
            "out_layer": [output_size],
        }
    elif local:
        model_parameters["num_bus"] = train_set.dataset[0].x.size()[0]
        model_parameters["num_output_cols"] = 2
        if architecture in ["gcn", "gat"]:
            if model_parameters["architecture_size"] == "big":
                model_parameters["gnn_layers"] = {
                    "0": [train_set.dataset[0].x.size()[1], 8],
                    "1": [8, max(8, train_set.dataset[0].x.size()[0])],
                    "2": [max(8, train_set.dataset[0].x.size()[0]), num_output*2],
                    "3": [num_output*2, 2],
                }
            elif model_parameters["architecture_size"] == "small":
                model_parameters["gnn_layers"] = {
                    "0": [train_set.dataset[0].x.size()[1], 8],
                    "1": [8, max(8, train_set.dataset[0].x.size()[0])],
                    "2": [max(8, train_set.dataset[0].x.size()[0]), num_output],
                    "3": [num_output, 2],
                }
        if architecture in ["chnn", "snn"]:
            if model_parameters["architecture_size"] == "big":
                model_parameters["gnn_layers"] = {
                    "0": [train_set.dataset[0].x.size()[1], 8],
                    "1": [8, max(8, int(np.floor(train_set.dataset[0].x.size()[0]/2)))],
                    "2": [max(8, int(np.floor(train_set.dataset[0].x.size()[0]/2))), int(np.floor(num_output))],
                    "3": [int(np.floor(num_output)), 2],
                }
            elif model_parameters["architecture_size"] == "small":
                model_parameters["gnn_layers"] = {
                   "0": [train_set.dataset[0].x.size()[1], 8],
                    "1": [8, max(8, int(np.floor(train_set.dataset[0].x.size()[0]/2)))],
                    "2": [max(8, int(np.floor(train_set.dataset[0].x.size()[0]/2))), int(np.floor(num_output/2))],
                    "3": [int(np.floor(num_output/2)), 2],
            }
    else:
        model_parameters["num_bus"] = train_set.dataset[0].x.size()[0]
        model_parameters["num_output_cols"] = 2
        if model_parameters["architecture_size"] == "big":
            model_parameters["gnn_layers"] = {
                "0": [train_set.dataset[0].x.size()[1], 16],
                "1": [16, 32],
                "2": [32, 2],
            }
        elif model_parameters["architecture_size"] == "small":
            model_parameters["gnn_layers"] = {
                "0": [train_set.dataset[0].x.size()[1], 8],
                "1": [8, 16],
                "2": [16, 1],
            }
        output_size = (
            num_output
            if primal
            else train_set.dataset[0].y.size(0)
        )
        model_parameters["fcnn_layers"] = fcnn_layers(
            model_parameters["gnn_layers"][
                list(model_parameters["gnn_layers"].keys())[-1]
            ][1]
            * model_parameters["num_bus"],
            output_size,
            1,
        )

    return model_parameters


class MeanSquaredError(nn.Module):
    def __init__(self, device, local, mask):
        super().__init__()
        self.device = device
        self.local = local
        self.mask = mask

    def forward(self, y_pred, y_true):
        if self.local:
            return ((y_pred[:, self.mask] - y_true.to(self.device)[:, self.mask]) ** 2).mean()
        else:
            return ((y_pred - y_true.to(self.device)[:, self.mask]) ** 2).mean()


def loss_function(device, primal=True, local=True, positive_wt=None, mask=None):
    if primal:
        return MeanSquaredError(device, local, mask)
    else:
        return nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=torch.tensor(positive_wt)
        )


def layer_size(layer, num_layers, size_in, size_out):
    return floor(size_in + ((layer) / (num_layers)) * (size_out - size_in))


def fcnn_layers(size_in, size_out, num_layers):
    layers = {}
    for l in range(num_layers):
        layers[l] = [
            layer_size(l, num_layers, size_in, size_out),
            layer_size(l + 1, num_layers, size_in, size_out),
        ]
    return layers


def create_mini_batches(data, batch_size, shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    mini_batches = []
    num_mini_batches = len(data) // batch_size
    for i in range(num_mini_batches + 1):
        mini_batch = data[i * batch_size : (i + 1) * batch_size]
        mini_batches.append(mini_batch)
    return mini_batches[:-1]


def initialise_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr)


def compile_model(parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = parameters["architecture"]
    if (architecture == "fcnn1") | (architecture == "fcnn"):
        model = models.FCNN(parameters).to(device)
    elif architecture == "cnn":
        model = models.CNN(parameters).to(device)
    else:
        model = models.GNN(parameters).to(device)
    return model, device


def check_patience(loss, patience_remaining, patience_start):
    val_loss_latest = np.hsplit(np.array(loss), 2)[1][-1]
    val_loss_best = np.min(np.hsplit(np.array(loss), 2)[1][:-1])
    if val_loss_latest < val_loss_best:
        patience_remaining = patience_start
    else:
        patience_remaining -= 1
    return patience_remaining


def update_weights(
    train_set,
    model,
    objective,
    optimizer,
    device,
    architecture,
    primal,
    local,
    batch_size,
    num_output_cols,
):
    model.train()
    if architecture in ["fcnn", "fcnn1", "cnn"]:
        for X_batch, Y_batch in train_set:
            optimizer.zero_grad()
            Y_pred = model(X_batch.to(device))
            Y_true = Y_batch.to(device)
            objective(Y_pred, Y_true).backward()
            optimizer.step()
    else:
        for batch in train_set:
            optimizer.zero_grad()
            batch = batch.to(device)
            if local & primal:
                Y_pred = model(batch).reshape(batch_size, -1, num_output_cols)
            else:
                Y_pred = torch.stack(model(batch))
            Y_true = batch.y
            if primal:
                Y_true = Y_true.reshape(batch_size, -1, num_output_cols)
            else:
                Y_true = Y_true.reshape(batch_size, -1)
            objective(Y_pred, Y_true).backward()
            optimizer.step()


def evaluate(
    data_set,
    model,
    objective,
    device,
    architecture,
    primal,
    local,
    batch_size,
    num_output_cols,
    return_predictions=False,
):
    model.eval()
    with torch.no_grad():
        loss = []
        preds = []
        if architecture in ["fcnn", "fcnn1", "cnn"]:
            for X_batch, Y_batch in data_set:
                Y_pred = model(X_batch.to(device))
                Y_true = Y_batch.to(device)
                loss.append(objective(Y_pred, Y_true).item())
                preds.append(Y_pred)
        else:
            for batch in data_set:
                batch = batch.to(device)
                if local & primal:
                    Y_pred = model(batch).reshape(batch_size, -1, num_output_cols)
                else:
                    Y_pred = torch.stack(model(batch))
                Y_true = batch.y
                if primal:
                    Y_true = Y_true.reshape(batch_size, -1, num_output_cols)
                else:
                    Y_true = Y_true.reshape(batch_size, -1)
                loss.append(objective(Y_pred, Y_true).item())
                preds.append(Y_pred)
    loss = torch.mean(torch.tensor(loss)).item()
    if return_predictions:
        return loss, preds
    else:
        return loss


def train(
    train_set,
    val_set,
    model,
    objective,
    optimizer,
    device,
    architecture,
    patience_start,
    min_epochs,
    max_epochs,
    primal,
    local,
    batch_size,
    num_output_cols,
):
    epoch = 0
    do_stop = False
    running_loss = []
    patience_remaining = patience_start
    start_time = time.time()
    while not do_stop:
        update_weights(
            train_set,
            model,
            objective,
            optimizer,
            device,
            architecture,
            primal,
            local,
            batch_size,
            num_output_cols,
        )
        train_loss = evaluate(
            train_set,
            model,
            objective,
            device,
            architecture,
            primal,
            local,
            batch_size,
            num_output_cols,
        )
        val_loss = evaluate(
            val_set,
            model,
            objective,
            device,
            architecture,
            primal,
            local,
            batch_size,
            num_output_cols,
        )
        running_loss.append([train_loss, val_loss])
        if epoch % 10 == 0:
            info = "$$$ Epoch: %-3d" % epoch
            info += "; train_loss: %8.5f" % train_loss
            info += "; val_loss: %8.5f" % val_loss
            info += "; patience_remaining: %d" % patience_remaining
            logger.info(info)
        epoch += 1
        if epoch > min_epochs:
            patience_remaining = check_patience(
                running_loss, patience_remaining, patience_start
            )
            if (patience_remaining < 0) | (epoch >= max_epochs):
                do_stop = True

    end_time = time.time()
    if patience_remaining < 0:
        logger.info(f"$$$ Patience exhausted after {epoch} epochs.")
    elif epoch >= max_epochs:
        logger.info(f"$$$ Maximum number of epochs ({max_epochs}) exceeded.")
    return running_loss, epoch, end_time - start_time
