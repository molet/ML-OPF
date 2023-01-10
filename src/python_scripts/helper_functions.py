### Import libraries.
import json
import numpy as np
import torch
import torch.nn as nn

### Import models.
import models as m

### Specific functions.
from collections import Counter
from math import floor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import add_self_loops

### Class to prepare data for dataloader.
class DataSetConstructor(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
  def __getitem__(self, index):
    return  (self.X[index], self.Y[index])
  def __len__(self):
    return len(self.X)

### Function to vectorise dictionary MathOptInterface constraints.
def binding_to_classifier(constraint_dict):
  AC_INEQUALITY_CONSTRAINTS = [
    "(VariableRef, MathOptInterface.GreaterThan{Float64})",
    "(VariableRef, MathOptInterface.LessThan{Float64})",
    "(GenericAffExpr{Float64,VariableRef}, MathOptInterface.GreaterThan{Float64})",
    "(GenericAffExpr{Float64,VariableRef}, MathOptInterface.LessThan{Float64})",
    "(GenericQuadExpr{Float64,VariableRef}, MathOptInterface.LessThan{Float64})"]
  clf_format = []
  for constraint_type in AC_INEQUALITY_CONSTRAINTS:
    clf_format += list(constraint_dict[constraint_type])
  return clf_format

### Function to retrieve set of non-trivial constraints.
def NonTrivialConstraints(Y):
  ### Determine the binding frequency of each constraint.
  binding_count = Counter({})
  for y in Y:
    for constraint_type in y.keys():
      y[constraint_type] = np.array(list(map(int, y[constraint_type])))
    binding_count.update(Counter(y))
  ### Flag constraints whuch are either never binding or always binding.
  non_trivial = {}
  for constraint_type in binding_count.keys():
    non_trivial[constraint_type] = (np.array(binding_count[constraint_type]) > 0) & (np.array(binding_count[constraint_type]) < len(Y))
  return non_trivial

### Function to parse raw data.
def ParseData(exp_parameters):

  ffnet, convnet1d, convnet2d, graphnet = [model in exp_parameters["model"] for model in ["ffnet", "convnet1d", "convnet2d", "graphnet"]]
  if graphnet & (exp_parameters["graph_layer"] == "None"):
    raise ValueError("Error: No graph layer assigned.")

  with open(exp_parameters["json_path"]) as src:
    raw_data = json.load(src)

  bus_params = [p for p in exp_parameters["bus"].keys() if exp_parameters["bus"][p] == True]
  line_params = [p for p in exp_parameters["line"].keys() if exp_parameters["line"][p] == True]

  n_samples = len(raw_data["adj_mat"])
  n_bus = len(raw_data["adj_mat"][0])
  n_params = len(bus_params) + len(line_params)

  if exp_parameters["primal"]:
    Y = [torch.tensor(yp, dtype=torch.float) for yp in raw_data["y_primal"]]
  else:
    Y = [yr for yr in raw_data["y_regime"]]

  if ffnet | convnet1d | convnet2d:
    if ffnet:
      X = [torch.cat([torch.tensor(x[p]) for p in bus_params + line_params]) for x in raw_data["x_vector"]]
    else:
      X = []
      for i in range(n_samples):
        if convnet1d:
          if len(line_params) > 0:
            raise ValueError("Error: ConvNet1d not compatible with line parameters.")
          x = torch.zeros(len(bus_params), n_bus)
          for j, p in enumerate(bus_params):
            x[j, :] = torch.tensor(raw_data["node_attr"][i][p], dtype=torch.float)
        else: # convnet2d.
          x = torch.zeros(n_params, n_bus, n_bus)
          for j, p in enumerate(bus_params):
            x[j, :, :].diagonal().copy_(torch.tensor(raw_data["node_attr"][i][p], dtype=torch.float))
          for j, p in enumerate(line_params):
            x[len(bus_params)+j, :, :] = torch.tensor(raw_data["edge_attr"][i][p], dtype=torch.float)
        X.append(x)

    ### Tran-Val-Test split (80/10/10)
    X_train, X_val, X_test = X[0:round(n_samples * 0.8)], X[round(n_samples * 0.8):round(n_samples * 0.9)], X[round(n_samples * 0.9):n_samples]
    Y_train, Y_val, Y_test = Y[0:round(n_samples * 0.8)], Y[round(n_samples * 0.8):round(n_samples * 0.9)], Y[round(n_samples * 0.9):n_samples]
    ### Non-trivial constraints.
    if not(exp_parameters["primal"]):
      non_trivial_train = NonTrivialConstraints(Y_train)
      Y_train = [torch.tensor(list(map(int, binding_to_classifier(y))), dtype=torch.float)[binding_to_classifier(non_trivial_train)] for y in Y_train]
      Y_val = [torch.tensor(list(map(int, binding_to_classifier(y))), dtype=torch.float)[binding_to_classifier(non_trivial_train)] for y in Y_val]
      Y_test = [torch.tensor(list(map(int, binding_to_classifier(y))), dtype=torch.float)[binding_to_classifier(non_trivial_train)] for y in Y_test]
    ### Pass data to dataloader.
    batch_size = round((len(X_train) + len(X_val) + len(X_test)) / 100)
    train_set = DataLoader(dataset=DataSetConstructor(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_set = DataLoader(dataset=DataSetConstructor(X_val, Y_val), batch_size=batch_size, shuffle=False)
    test_set = DataLoader(dataset=DataSetConstructor(X_test, Y_test), batch_size=len(X_test), shuffle=False)

  else: # graphnet

    X = [torch.cat([torch.tensor(n[p], dtype=torch.float).unsqueeze(dim=1) for p in bus_params] + [torch.tensor(e[p], dtype=torch.float) for p in line_params], dim=1) for n, e in zip(raw_data["node_attr"], raw_data['edge_attr'])]
    ### Self loops.
    edge_index = [add_self_loops(dense_to_sparse(torch.tensor(am))[0], dense_to_sparse(torch.tensor(am))[1])[0] for am in raw_data["adj_mat"]]
    edge_attr = [add_self_loops(dense_to_sparse(torch.tensor(am))[0], dense_to_sparse(torch.tensor(am))[1])[1].unsqueeze(1) for am in raw_data["adj_mat"]]
    ### No self loops.
    #edge_index = [dense_to_sparse(torch.tensor(am))[0], dense_to_sparse(torch.tensor(am))[1][0] for am in data["adj_mat"]]
    #edge_attr = [dense_to_sparse(torch.tensor(am))[0], dense_to_sparse(torch.tensor(am))[1][1].unsqueeze(1) for am in data["adj_mat"]]
    data = [Data(x=x, edge_index=ei, edge_attr=ea, y=y) for x, ei, ea, y in zip(X, edge_index, edge_attr, Y)]
    ### Tran-Val-Test split (80/10/10)
    train_set, val_set, test_set = data[0:round(n_samples * 0.8)], data[round(n_samples * 0.8):round(n_samples * 0.9)], data[round(n_samples * 0.9):n_samples]
    if not(exp_parameters["primal"]):
      non_trivial_train = NonTrivialConstraints([d.y for d in train_set])
      for d in train_set:
        d.y = torch.tensor(list(map(int, binding_to_classifier(d.y))), dtype=torch.float)[binding_to_classifier(non_trivial_train)]
      for d in val_set:
        d.y = torch.tensor(list(map(int, binding_to_classifier(d.y))), dtype=torch.float)[binding_to_classifier(non_trivial_train)]
      for d in test_set:
        d.y = torch.tensor(list(map(int, binding_to_classifier(d.y))), dtype=torch.float)[binding_to_classifier(non_trivial_train)]

  return train_set, val_set, test_set

### Convolutional layer output size.
def ConvDim(input_size, n_conv_layers, k=3, p=0, s=1, mp=2):
  for i in range(n_conv_layers):
    output_size = (1 + (input_size - k + 2 * p) / s) / mp
    input_size = output_size
  return floor(output_size)

### Function to construct mini batches.
def CreateMiniBatches(data, batch_size):
    mini_batches = []
    np.random.shuffle(data)
    n_minibatches = len(data) // batch_size
    i = 0
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size]
        mini_batches.append(mini_batch)
    return mini_batches[:-1]

### Define Loss Function.
def loss_function(primal=True):
  if primal:
    l = nn.MSELoss(reduction="mean")
  else:
    l = nn.BCELoss(reduction="mean")
  return l

### Pass model to device.
def CreateModel(model_parameters, exp_parameters):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ffnet, convnet1d, convnet2d, graphnet = [model in exp_parameters["model"] for model in ["ffnet", "convnet1d", "convnet2d", "graphnet"]]
  gcnconv, chebconv, splineconv = [layer in exp_parameters["graph_layer"] for layer in ["gcnconv", "chebconv", "splineconv"]]
  if ffnet:
    model = m.FFNet(model_parameters).to(device)
  elif convnet1d:
    model = m.ConvNet1d(model_parameters).to(device)
  elif convnet2d:
    model = m.ConvNet2d(model_parameters).to(device)
  else: # graphnet.
    if gcnconv:
      model = m.GCN(model_parameters).to(device)
    elif chebconv:
      model = m.ChebNet(model_parameters).to(device)
    else: # splinenet.
      model = m.SplineNet(model_parameters).to(device)
  return model, device

### Training the model.
def train(
  model,
  train_set,
  val_set,
  objective,
  optimizer,
  device,
  graphnet):

  if graphnet:
    batch_size = round(len(val_set) / 10)
    train_mini_batches = CreateMiniBatches(train_set, batch_size)
    val_mini_batches = CreateMiniBatches(val_set, batch_size)

  ### Initalise training sequence.
  model.train()
  if graphnet:
    for batch in train_mini_batches:
      optimizer.zero_grad()
      torch.stack([objective(y_hat, y.to(device)) for y_hat, y in zip(model(batch), [data.y for data in batch])]).mean().backward()
      optimizer.step()
  else:
    for X_batch, Y_batch in train_set:
      optimizer.zero_grad()
      X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
      objective(model(X_batch), Y_batch).backward()
      optimizer.step()

  ### Initalise model evalutation.
  model.eval()
  if graphnet:
    train_loss = []
    with torch.no_grad():
      for batch in train_mini_batches:
        train_loss.append(torch.stack([objective(y_hat, y.to(device)) for y_hat, y in zip(model(batch), [data.y for data in batch])]).mean().item())
    val_loss = []
    with torch.no_grad():
      for batch in val_mini_batches:
        val_loss.append(torch.stack([objective(y_hat, y.to(device)) for y_hat, y in zip(model(batch), [data.y for data in batch])]).mean().item())
  else:
    train_loss = []
    with torch.no_grad():
      for X_batch, Y_batch in train_set:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        train_loss.append(objective(model(X_batch), Y_batch).item())
    val_loss = []
    with torch.no_grad():
      for X_batch, Y_batch in val_set:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        val_loss.append(objective(model(X_batch), Y_batch).item())

  return torch.mean(torch.tensor(train_loss)).item(), torch.mean(torch.tensor(val_loss)).item()
  
def check_patience(loss, remaining_patience, patience):
    l_val = np.hsplit(np.array(loss), 2)[1]
    latest_loss = l_val[-1]
    best_loss = np.min(l_val[:-1])
    if latest_loss < best_loss:
        remaining_patience = patience
    else:
        remaining_patience -= 1
    return remaining_patience

### Test the model.
def test(
  model,
  test_set,
  objective,
  device,
  graphnet):

  if graphnet:
    batch_size = round(len(test_set) / 10)
    test_mini_batches = CreateMiniBatches(test_set, batch_size)

  ### Initalise model evaluation.
  model.eval()
  with torch.no_grad():
    test_loss = []
    Y_hat = []
    if graphnet:
      for batch in test_mini_batches:
        Y_hat_batch = model(batch)
        test_loss.append(torch.stack([objective(y_hat, y.to(device)) for y_hat, y in zip(model(batch), [data.y for data in batch])]).mean().item())
        Y_hat.append(Y_hat_batch)
    else:
      for X_batch, Y_batch in test_set:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        Y_hat_batch = model(X_batch)
        test_loss.append(objective(Y_hat_batch, Y_batch).item())
        Y_hat.append(Y_hat_batch)
  if graphnet:
    return torch.mean(torch.tensor(test_loss)).item(), Y_hat
  else:
    return torch.mean(torch.tensor(test_loss)).item(), Y_hat[0].cpu().data.numpy()

  
def SaveJSON(results, path):
    with open(path, 'w') as src:
        json.dump(results, src)
