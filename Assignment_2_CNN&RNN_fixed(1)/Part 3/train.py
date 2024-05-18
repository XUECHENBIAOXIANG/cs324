from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy


def train(model, data_loader, optimizer, criterion, device, config):
    # TODO set model to train mode
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        inputs, targets = batch_inputs.to(device), batch_targets.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, targets)

        loss.backward()

        # 更新参数
        optimizer.step()
        # 计算准确率
        acc = accuracy(outputs, targets)
        losses.update(loss.item())
        accuracies.update(acc)

        # Add more code here ...
        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here     ...
        inputs, targets = batch_inputs.to(device), batch_targets.to(device)
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, targets)
        # 计算准确率
        acc = accuracy(outputs, targets)
        losses.update(loss.item())
        accuracies.update(acc)
        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the model that we are going to use
    model = VanillaRNN(input_length=config.input_length,
                       input_dim=config.input_dim,
                       hidden_dim=config.num_hidden,
                       output_dim=config.num_classes,
                       device=device)  # fixme
    model.to(device)

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(
        input_length=config.input_length, total_len=config.data_size)

    # fixme
    # Split dataset into train and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [config.portion_train,
                                                                         1 - config.portion_train])  # fixme
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)  # fixme
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)  # fixme

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # fixme
    train_loss_data, train_acc_data,test_loss_data,test_acc_data = [], [], [], []

    for epoch in range(config.max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config)

        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device, config)
        scheduler.step()
        print(
            f'Epoch: {epoch + 1}/{config.max_epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val. Loss: {val_loss:.4f}, Val. Acc: {val_acc:.4f}')
        train_loss_data.append(train_loss)
        train_acc_data.append(train_acc)
        test_loss_data.append(val_loss)
        test_acc_data.append(val_acc)
    print('Done training.')
    return train_loss_data, train_acc_data, test_loss_data, test_acc_data


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=6,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=1000, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=10000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)
