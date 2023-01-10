import sys
import argparse
import json
import math
import time

import numpy as np
import torch

from log import getLogger
import data_parser as parser
import model_utils as utils
import models

logger = getLogger(__name__)

def run_experiment(args):

    logger.info("setting parameters.")
    args = args.copy()
    grid = args.pop("grid")
    learning_rate = args.pop("learning_rate")
    patience = args.pop("patience")
    min_epochs = args.pop("min_epochs")
    max_epochs = args.pop("max_epochs")

    experiment_parameters = args
    args.update({"features": ["pload", "qload"]})

    logger.info(f'grid: {grid}')
    logger.info(f'primal: {experiment_parameters["primal"]}')
    logger.info(f'local: {experiment_parameters["local"]}')
    logger.info(f'batch size: {experiment_parameters["batch_size"]}')
    logger.info(f'seed: {experiment_parameters["seed"]}')
    logger.info(f'architecture: {experiment_parameters["architecture"]}')
    logger.info(f'architecture_size: {experiment_parameters["architecture_size"]}')

    parsed_data = parser.DataParser(experiment_parameters)
    model_parameters = utils.construct_model_parameters(
        parsed_data.train_set,
        parsed_data.architecture,
        parsed_data.architecture_size,
        parsed_data.batch_size,
        parsed_data.primal,
        parsed_data.kernel_size,
        parsed_data.local,
        parsed_data.gen_bus_indicies.size(0),
        int(parsed_data.mask.sum())
    )

    model, device = utils.compile_model(model_parameters)
    logger.info(f"device: {device}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable parameters: {trainable_params}")

    optimizer = utils.initialise_optimizer(model, learning_rate)
    objective = utils.loss_function(
        device,
        parsed_data.primal,
        parsed_data.local,
        positive_wt=parsed_data.positive_wt,
        mask=parsed_data.mask,
    )

    running_loss, epoch, elapsed_time = utils.train(
        parsed_data.train_set,
        parsed_data.val_set,
        model,
        objective,
        optimizer,
        device,
        parsed_data.architecture,
        patience,
        min_epochs,
        max_epochs,
        parsed_data.primal,
        parsed_data.local,
        parsed_data.batch_size,
        model_parameters["num_output_cols"],
    )

    test_loss, test_preds = utils.evaluate(
        parsed_data.test_set,
        model,
        objective,
        device,
        parsed_data.architecture,
        parsed_data.primal,
        parsed_data.local,
        parsed_data.batch_size,
        model_parameters["num_output_cols"],
        return_predictions=True,
    )

#   if parsed_data.primal:
#       test_set = [parsed_data.primals[i] for i in parsed_data.indicies_test]
#       test_set = torch.tensor(np.array(test_set, dtype=np.float))
#       test_preds = [test_pred.T for test_pred in test_preds]
#       test_preds = torch.cat([torch.stack([test_pred[:, i] for i in range(parsed_data.batch_size)]) for test_pred in test_preds])
#       test_preds = parser.input_predictions(test_preds, test_set, parsed_data.batch_size)        
#       test_preds = parser.reverse_normalise(  # Reverse normalise predictions.
#           test_preds.numpy(), parsed_data.Y_min, parsed_data.Y_max
#       )

#   else:
#       if parsed_data.architecture in ["fcnn", "fcnn1", "cnn"]:
#           test_set = torch.tensor(
#               [y.cpu().detach().numpy() for y in parsed_data.test_set.dataset.Y]
#           )
#       else:
#           test_set = torch.tensor(
#               [y.cpu().detach().numpy() for y in parsed_data.test_set.dataset]
#           )

#   if parsed_data.architecture in ["fcnn","fcnn1","cnn"]:  # Make sets serializable.
#       test_preds = [test_preds[i].tolist() for i in range(len(test_preds))]
#       test_set = test_set.numpy().tolist()

#   else:
#       if parsed_data.architecture in ["fcnn", "fcnn1", "cnn"]:
#           test_preds = np.array(test_preds).tolist()
#       else:
#           if torch.is_tensor(test_preds):
#               test_preds = test_preds.cpu().detach().numpy().tolist()
#           else:
#               test_preds = np.array([torch.tensor(test_pred).cpu().detach().numpy() for test_pred in test_preds]).tolist()
#       
#       test_set = test_set.numpy().tolist()

    results = {
        "primal": parsed_data.primal,
        "total_epochs": epoch,
        "trainable_params": trainable_params,
        "elapsed_time": elapsed_time,
        "running_loss": running_loss,
        "test_loss": test_loss,
#       "test_preds": test_preds,
#       "test_set": test_set,
    }
    
    prefix = "reg_local_" if (parsed_data.primal & parsed_data.local) else ("reg_global_" if parsed_data.primal else "clf_global_")
    filename = prefix + parsed_data.architecture + f"_{parsed_data.seed}" + "_case_" + grid + ".json"
    parser.save_json(results, parsed_data.results + filename)
    logger.info(f"finished: case => {grid}; seed => {parsed_data.seed}; arch => {parsed_data.architecture}")
    logger.info(f"output: saved @ {filename}")
