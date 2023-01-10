from collections import Counter
import json

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, DataLoader as GraphDataLoader
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import add_self_loops
import torch
import numpy as np

AC_INEQUALITY_CONSTRAINTS = [
    "(JuMP.VariableRef, MathOptInterface.GreaterThan{Float64})",
    "(JuMP.VariableRef, MathOptInterface.LessThan{Float64})",
    "(JuMP.GenericAffExpr{Float64,JuMP.VariableRef}, MathOptInterface.GreaterThan{Float64})",
    "(JuMP.GenericAffExpr{Float64,JuMP.VariableRef}, MathOptInterface.LessThan{Float64})",
    "(JuMP.GenericQuadExpr{Float64,JuMP.VariableRef}, MathOptInterface.LessThan{Float64})",
]


class DatasetConstructor(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return len(self.X)


def save_json(dict, path):
    with open(path, "w") as src:
        json.dump(dict, src)


def load_json(path):
    with open(path) as src:
        return json.load(src)


def binding_to_classifier(constraint_dict, labels=AC_INEQUALITY_CONSTRAINTS):
    clf_format = []
    for constraint_type in labels:
        clf_format += list(constraint_dict[constraint_type])
    return clf_format


def non_trivial_constraints(Y):
  binding_freq = Counter({})
  for y in Y:
    for constraint_type in y.keys():
      y[constraint_type] = np.array(list(map(int, y[constraint_type])))
    binding_freq.update(Counter(y))
  non_trivial = {}
  for constraint_type in binding_freq.keys():
    non_trivial[constraint_type] = (np.array(binding_freq[constraint_type]) > 0) & (np.array(binding_freq[constraint_type]) < len(Y))
  return non_trivial

def select_primals(Y, columns=(4, 7)):
    return [[Y[i][column] for column in columns] for i in range(len(Y))]

def get_min_max(Y):
    stacked = np.vstack(
        [np.hstack([Y[i][j] for j in range(len(Y[i]))]) for i in range(len(Y))]
    )
    return np.min(stacked, axis=0), np.max(stacked, axis=0)


def normalise(Y, Y_min, Y_max):
    Y = np.array(Y.copy(), dtype=np.float)
    for i in range(len(Y)):
        Y_stack = np.hstack([Y[i][j] for j in range(len(Y[i]))])
        with np.errstate(divide="ignore", invalid="ignore"):
            Y_norm = np.nan_to_num((Y_stack - Y_min) / (Y_max - Y_min)).reshape(
                -1, len(Y[i]), order="F"
            )
            for j in range(len(Y[i])):
                Y[i][j] = Y_norm[:, j]
    return Y


def reverse_normalise(Y, Y_min, Y_max):
    Y = Y.copy()
    for i in range(len(Y)):
        Y_stack = np.hstack([Y[i][j] for j in range(len(Y[i]))])
        Y_norm_reverse = ((Y_stack * (Y_max - Y_min)) + Y_min).reshape(
            -1, len(Y[i]), order="F"
        )
        for j in range(len(Y[i])):
            Y[i][j] = Y_norm_reverse[:, j]
    return Y

def to_sparse_matrix(mat):
    return dense_to_sparse(torch.tensor(mat, dtype=torch.float))


def input_predictions(test_preds, test_set, batch_size):
    _test_preds = test_set.clone()
    for i in range(batch_size):
        _test_preds[i][4] = test_preds[i][0]
        _test_preds[i][7] = test_preds[i][1]  
    return _test_preds


class DatasetConstructor(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return len(self.X)


class DataParser:
    def __init__(self, parameters):

        self.parameters = parameters
        for key in self.parameters:
            setattr(self, key, self.parameters[key])
        self.raw_data = load_json(self.samples)
        for key in self.raw_data:
            setattr(self, key, self.raw_data[key])
        self.num_samples = len(self.node_attributes)
        self.num_features = len(self.features)
        self.num_bus = len(self.adj_mat[0])
        self.gen_bus_mask = ~torch.tensor(
            np.array(self.primals[0][3], dtype=np.float64)).isnan()
        self.gen_bus_indicies = torch.tensor(self.primals[0][1])
        self.num_gen = len(self.gen_bus_indicies[self.gen_bus_mask]) 
        self.imp_mat = self._update_imp_mat()
        self.imp_index, self.imp_values = to_sparse_matrix(self.imp_mat)
        self.imp_values = np.exp(-self.kernel_wt * self.imp_values)
        self.positive_wt = None
        self._set_indicies()
        if self.architecture in ["fcnn", "fcnn1"]:
            self._parse_fcnn_set(self._parse_fcnn_input())

        elif self.architecture == "cnn":
            self._parse_cnn_set()

        else:
            self._parse_gnn_set()
        self.mask = self.get_mask()

    def _update_imp_mat(self):
        n = len(self.gen_bus_indicies)
        imp_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if self.gen_bus_indicies[i] == self.gen_bus_indicies[j]:
                    imp_mat[i, j] = 1.0
                else:
                    imp_mat[i, j] = self.imp_mat[self.gen_bus_indicies[i] - 1][self.gen_bus_indicies[j] - 1]
                imp_mat[j, i] = imp_mat[i, j]
        return imp_mat

    def _parse_fcnn_input(self):
        X = []
        for x in self.node_attributes:
            features = []
            for feature in self.features:
                features.append(torch.tensor(x[feature], dtype=torch.float))
            X.append(torch.cat(features))
        return X

    def _parse_cnn_input(self):
        X = []
        for i in range(self.num_samples):
            x = torch.zeros(self.num_features, self.num_bus)
            for j, feature in enumerate(self.features):
                x[j, :] = torch.tensor(
                    self.node_attributes[i][feature], dtype=torch.float
                )
            X.append(x)
        return X

    def _parse_gnn_input(self):
        X = []
        edge_index = []
        edge_attr = []
        for x in self.node_attributes:
            features = []
            for feature in self.features:
                features.append(
                    torch.tensor(x[feature], dtype=torch.float).unsqueeze(dim=1)
                )
            X.append(torch.cat(features, dim=1))
        for i in range(self.num_samples):
            edge_index.append(add_self_loops(self.imp_index, self.imp_values)[0])
            if self.architecture == "snn":
                edge_attr.append(
                    add_self_loops(self.imp_index, self.imp_values)[1].unsqueeze(1)
                )
            else:
                edge_attr.append(add_self_loops(self.imp_index, self.imp_values)[1])
        X = [self.augment_input(x) for x in X]
        return X, edge_index, edge_attr

    def augment_input(self, input):
        i = 0
        gen_bus = self.gen_bus_indicies.unique()
        num_bus_prime = len(self.gen_bus_indicies)
        input_prime = torch.zeros(num_bus_prime, input.shape[1])
        for bus in range(self.num_bus):
            num_gen = (self.gen_bus_indicies - 1 == bus).sum()
            if num_gen > 0:
                input_prime[i:i+num_gen] = input[bus].repeat(num_gen).reshape(-1, input.shape[1]) / num_gen
                i += num_gen
        return input_prime

    def _parse_primals(self):
        Y = select_primals(self.primals)
        self.Y_min, self.Y_max = get_min_max(Y)
        Y = normalise(Y, self.Y_min, self.Y_max)
        Y = [torch.tensor(y).T for y in Y]
        return Y

    def _parse_regime(self):
        return [y for y in self.regime]

    def _set_indicies(self):
        np.random.seed(self.seed)
        indicies = np.random.permutation(self.num_samples)
        self.indicies_train = indicies[0 : round(self.num_samples * self.train_size)]
        self.indicies_val = indicies[
            round(self.num_samples * self.train_size) : round(
                self.num_samples * (1 - self.test_size)
            )
        ]
        self.indicies_test = indicies[
            round(self.num_samples * (1 - self.test_size)) : self.num_samples
        ]

    def _filter_fcnn_nt_cons(self, Y_train, Y_val, Y_test):
        nt_train = non_trivial_constraints(Y_train)
        for i, y in enumerate(Y_train):
            Y_train[i] = list(map(int, binding_to_classifier(y)))
            Y_train[i] = torch.tensor(Y_train[i], dtype=torch.float)
            Y_train[i] = Y_train[i][binding_to_classifier(nt_train)]
        for i, y in enumerate(Y_val):
            Y_val[i] = list(map(int, binding_to_classifier(y)))
            Y_val[i] = torch.tensor(Y_val[i], dtype=torch.float)
            Y_val[i] = Y_val[i][binding_to_classifier(nt_train)]
        for i, y in enumerate(Y_test):
            Y_test[i] = list(map(int, binding_to_classifier(y)))
            Y_test[i] = torch.tensor(Y_test[i], dtype=torch.float)
            Y_test[i] = Y_test[i][binding_to_classifier(nt_train)]
        self.positive_wt = torch.stack(Y_train, axis=0).sum(axis=0) / len(Y_train)
        self.positive_wt = self.positive_wt.mean().numpy().item()
        return Y_train, Y_val, Y_test

    def _filter_gnn_nt_cons(self, train_set, val_set, test_set):
        nt_train = non_trivial_constraints([row.y for row in train_set])
        for row in train_set:
            row.y = list(map(int, binding_to_classifier(row.y)))
            row.y = torch.tensor(row.y, dtype=torch.float)
            row.y = row.y[binding_to_classifier(nt_train)]
        for row in val_set:
            row.y = list(map(int, binding_to_classifier(row.y)))
            row.y = torch.tensor(row.y, dtype=torch.float)
            row.y = row.y[binding_to_classifier(nt_train)]
        for row in test_set:
            row.y = list(map(int, binding_to_classifier(row.y)))
            row.y = torch.tensor(row.y, dtype=torch.float)
            row.y = row.y[binding_to_classifier(nt_train)]
        self.positive_wt = torch.stack([d.y for d in train_set], axis=0).sum(axis=0) / len(train_set)
        self.positive_wt  = self.positive_wt.mean().numpy().item()
        return train_set, val_set, test_set


    def _parse_output(self):
        if self.primal:
            Y = self._parse_primals()
        else:
            Y = self._parse_regime()
        return Y

    def _parse_fcnn_set(self, X):
        Y = self._parse_output()
        X_train, X_val, X_test = (
            [X[i] for i in self.indicies_train],
            [X[i] for i in self.indicies_val],
            [X[i] for i in self.indicies_test],
        )
        Y_train, Y_val, Y_test = (
            [Y[i] for i in self.indicies_train],
            [Y[i] for i in self.indicies_val],
            [Y[i] for i in self.indicies_test],
        )

        if not (self.primal):
            Y_train, Y_val, Y_test = self._filter_fcnn_nt_cons(Y_train, Y_val, Y_test)

        self.train_set = DataLoader(
            dataset=DatasetConstructor(X_train, Y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_set = DataLoader(
            dataset=DatasetConstructor(X_val, Y_val),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.test_set = DataLoader(
            dataset=DatasetConstructor(X_test, Y_test),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def _parse_cnn_set(self):
        self._parse_fcnn_set(self._parse_cnn_input())

    def _parse_gnn_set(self):
        X, edge_index, edge_attr = self._parse_gnn_input()
        Y = self._parse_output()
        data_set = []
        for x, y, ei, ea in zip(X, Y, edge_index, edge_attr):
            data_set.append(Data(x=x, y=y, edge_index=ei, edge_attr=ea))
        self.train_set = [data_set[i] for i in self.indicies_train]
        self.val_set = [data_set[i] for i in self.indicies_val]
        self.test_set = [data_set[i] for i in self.indicies_test]
        if not (self.primal):
            data_set = self._filter_gnn_nt_cons(
                self.train_set, self.val_set, self.test_set
            )
            self.train_set, self.val_set, self.test_set = data_set

        self.train_set = GraphDataLoader(
            dataset=self.train_set, batch_size=self.batch_size, shuffle=True
        )
        self.val_set = GraphDataLoader(
            dataset=self.val_set, batch_size=self.batch_size, shuffle=False
        )
        self.test_set = GraphDataLoader(
            dataset=self.test_set, batch_size=self.batch_size, shuffle=False
        )


    def get_mask(self):
        i = 0
        gen_bus = self.gen_bus_indicies[self.gen_bus_mask].unique()
        num_bus_prime = self.gen_bus_indicies[self.gen_bus_mask].size(0) 
        num_bus_prime += np.sum([bus not in gen_bus for bus in torch.arange(1, self.num_bus+1)])
        mask = torch.zeros(num_bus_prime, 2)
        for bus in range(self.num_bus):
            num_gen = (self.gen_bus_indicies[self.gen_bus_mask] - 1 == bus).sum()
            if num_gen > 0:
                mask[i, 0] = 1
                mask[i:i+num_gen, 1] = torch.tensor([1]).repeat(num_gen)
                i += num_gen
            else:
                i += 1
        return mask.bool()
