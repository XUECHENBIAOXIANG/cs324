from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch

from cnn_model import CNN
from torch import nn, optim
from torchvision import datasets, transforms



# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './data'
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
    pred_classes = torch.argmax(predictions, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    return float(torch.mean((pred_classes == true_classes).float()))

def train(FLAGS):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=False)

    # Initialize the model
    model = CNN(n_channels=3, n_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize the criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Track loss and accuracy
    train_loss_history = []
    train_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Training loop
    for epoch in range(FLAGS.max_steps):
        model.train()
        running_loss = 0.0
        print(f'Epoch {epoch} started')

        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        model.eval()
        print('testing')
        with torch.no_grad():
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            test_accuracy = test_correct / test_total
            test_accuracy_history.append(test_accuracy)
            test_loss_history.append(test_loss / len(testloader))
            train_loss=0.0
            train_correct=0
            train_total=0
            for data in trainloader:
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            train_accuracy =train_correct/train_total
            train_accuracy_history.append(train_accuracy)
            train_loss_history.append(train_loss/len(trainloader))
        if epoch % FLAGS.eval_freq == 0:
            print(f'Epoch {epoch}, Train loss: {train_loss/len(trainloader)}, Train accuracy: {train_correct/train_total}')
            print(f'Epoch {epoch}, Test loss: {test_loss/len(testloader)}, Test accuracy: {test_accuracy}')




    print('Training finished')

    return train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history


def main():
    """
    Main function
    """
    train(FLAGS)

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()


  main()