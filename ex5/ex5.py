import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Adagrad
from torch.utils.data import random_split, TensorDataset, DataLoader
from data_loader import dataset, test_loader
from model import *
import train_file


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def select_activation(product, activation_name):
    if activation_name == "ReLu":
        return F.relu(product)
    elif activation_name == "sigmoid":
        return torch.sigmoid(product)
    elif activation_name == "softmax":
        return F.softmax(product)
    else:
        print("error")
        return None


def select_optimizer(model, optimizer_name, learning_rate):
    if optimizer_name == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        return Adagrad(model.parameters(), lr=learning_rate)
    else:
        print("error")
        return None


class FirstNet(nn.Module):
    def __init__(self, image_size, network_shape, activation="ReLu", optimizer="adam",
                 learning_rate=0.001, dropout=False, batch_norm=False):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        if torch.cuda.is_available():
            self.cuda()

        self.fc = nn.ModuleList([nn.Linear(image_size, network_shape[0])])
        for i in range(len(network_shape) - 1):
            self.fc.append(nn.Linear(network_shape[i], network_shape[i + 1]))
        if dropout:
            self.dropout_layer = nn.Dropout(p=0.25)
        if batch_norm:
            self.batch_layer = nn.ModuleList([nn.BatchNorm1d(item) for item in network_shape[:-1]])

        self.activation = activation
        self.optimizer = select_optimizer(self, optimizer, learning_rate)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = x.view(-1, self.image_size)
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            if hasattr(self, 'batch_layer'):
                x = self.batch_layer[i](x)
            x = select_activation(x, self.activation)
            if hasattr(self, 'dropout_layer'):
                x = self.dropout_layer(x)
        return F.log_softmax(x, dim=1)


def train_test_split(dataset, batch_size=64, train_size=0.9):
    train_length = round(train_size * len(dataset))
    test_length = len(dataset) - train_length
    train_set, val_set = random_split(dataset, [train_length, test_length])
    train_loader = DataLoader(train_set.dataset, batch_size=batch_size, shuffle=False)

    val_loader = DataLoader(val_set.dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def calc_accuracy(predictions, labels):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1
    return correct / len(predictions)


def train(model, train_set, val_set, epochs, model_name, device, batch_size, model_checkpoint=False):
    print(f"started training model {model_name}")
    model.train()
    avg_train_loss, avg_val_loss = [], []
    avg_train_acc, avg_val_acc = [], []
    max_valid_acc = 0

    for epoch in range(epochs):
        train_loss, valid_loss = 0, 0
        count_train_correct, count_val_correct = 0, 0
        for batch_idx, (data, labels) in enumerate(train_set):
            if torch.cuda.is_available():
                data, labels = data.to(device), labels.to(device)

            data = data.float()
            model.optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            model.optimizer.step()
            train_loss += loss.item()

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if torch.cuda.is_available():
                count_train_correct += pred.eq(labels.view_as(pred)).cuda().sum()
            else:
                count_train_correct += pred.eq(labels.view_as(pred)).cpu().sum()

        model.eval()  # Optional when not using Model Specific layer
        for batch_idx, (data, labels) in enumerate(val_set):
            if torch.cuda.is_available():
                data, labels = data.to(device), labels.to(device)

            data = data.float()
            output = model(data)
            loss = F.nll_loss(output, labels)
            valid_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            if torch.cuda.is_available():
                count_val_correct += pred.eq(labels.view_as(pred)).cuda().sum()
            else:
                count_val_correct += pred.eq(labels.view_as(pred)).cpu().sum()

        valid_loss, train_loss = valid_loss / len(val_set), train_loss / len(train_set)
        train_acc = count_train_correct / (batch_size * len(train_set))
        valid_acc = count_val_correct / (batch_size * len(val_set))
        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss}'
              f' \t\t Validation Loss: {valid_loss}')
        print(f'Training accuracy is: {train_acc} and '
              f'Validation accuracy is: {valid_acc}\n')

        if model_checkpoint and valid_acc > max_valid_acc and valid_acc > 0.9:
            max_valid_acc = valid_acc
            print("saving this model")
            PATH = f"models\\best_model{max_valid_acc}.pt"
            torch.save(model.state_dict(), PATH)

        avg_train_loss.append(train_loss)
        avg_val_loss.append(valid_loss)
        avg_train_acc.append(train_acc)
        avg_val_acc.append(valid_acc)

    plot_loss(avg_train_loss, avg_val_loss, model_name)
    plot_accuracy(avg_train_acc, avg_val_acc, model_name)


def plot_loss(avg_train_loss, avg_val_loss, model_name):
    plt.plot(avg_train_loss)
    plt.plot(avg_val_loss)
    plt.legend(["avg train loss", "avg val loss"])
    plt.title(f"train and validation loss for {model_name}")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(f'temp\\{model_name}_loss.png')
    plt.show()


def plot_accuracy(avg_train_acc, avg_val_acc, model_name):
    plt.plot(avg_train_acc)
    plt.plot(avg_val_acc)
    plt.legend(["avg train accuracy", "avg val accuracy"])
    plt.title(f"train and validation accuracy for {model_name}")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig(f'temp\\{model_name}_accuracy.png')
    plt.show()


def test(model, test_x, zero_test_y, batch_size=32):
    dataset = TensorDataset(test_x.clone().detach(),
                            zero_test_y.clone().detach())
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            data = data.float()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predictions += pred.tolist()

    predictions = [item[0] for item in predictions]

    return predictions


def log_results(y_hat, filename="test_y"):
    with open(filename, "w") as file_name:
        for i in range(len(y_hat) - 1):
            file_name.write(f"{y_hat[i]}\n")
        file_name.write(f"{y_hat[-1]}")


def main():
    device = get_default_device()
    epochs = 10
    batch_size = 1
    train_size = 0.9

    train_set, val_set = train_test_split(dataset, batch_size, train_size)
    print(f'finish loading data')
    # model_A = VGG('mini')
    # 520352 or 65044 or 16261
    model_A = FirstNet(16261, network_shape=(1000, 200, 30), optimizer='sgd', dropout=True)

    models = [model_A]
    letters = ["A"]

    for index, model in enumerate(models):
        print(f"started training model: {letters[index]}")
        model.to(device)
        train(model, train_set, val_set, epochs,
              "model_" + letters[index], device, batch_size, model_checkpoint=False)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'time took is: {end - start}')
