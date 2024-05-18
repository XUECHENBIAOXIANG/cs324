from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch
from torch import nn, optim

from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    # YOUR CODE HERE
    pred_classes = torch.argmax(predictions, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    return float(torch.mean((pred_classes == true_classes).float()))


def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, X_train, X_test, y_train, y_test):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    # Initialize the MLP model

    model = MLP(n_inputs=X_train.shape[1], n_hidden=[int(i) for i in dnn_hidden_units.split(',')],
                n_classes=y_train.shape[1])
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    CEL = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    accuracy_list = []
    accuracy_list_test = []
    loss_list = []
    loss_list_test = []
    for step in range(max_steps):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = CEL(y_pred, y)
            loss.backward()
            optimizer.step()
        # Calculate the loss and accuracy
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            loss = CEL(y_pred, y_train)
            loss_list.append(loss.item())
            accuracy_list.append(accuracy(y_pred, y_train))
            y_pred_test = model(X_test)
            loss_test = CEL(y_pred_test, y_test)
            loss_list_test.append(loss_test.item())
            accuracy_list_test.append(accuracy(y_pred_test, y_test))
        if step % eval_freq == 0:
            print('Step: ', step, 'train_Loss: ', loss.item(), 'train_Accuracy: ', accuracy(y_pred, y_train))
            print('Step: ', step, 'test_Loss: ', loss_test.item(), 'test_Accuracy: ', accuracy(y_pred_test, y_test))
    return accuracy_list, accuracy_list_test, loss_list, loss_list_test


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
