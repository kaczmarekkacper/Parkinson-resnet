from __future__ import print_function, division

import copy
import os.path
import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging


class ModelTrainer:
    def __init__(self, classes, filename):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.training_loss = list()
        self.training_accuracy = list()
        self.validation_loss = list()
        self.validation_accuracy = list()
        self.model = None
        self.classes = classes
        self.filename = filename

    def train_resnet(self, model, train_data_loader, test_data_loader, num_epochs):
        logging.debug(f'CPU/CUDA?: {self.device}')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.classes))
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

        self.training_loss = list()
        self.training_accuracy = list()
        self.validation_loss = list()
        self.validation_accuracy = list()

        self.model = self.train_model(train_data_loader, test_data_loader, model, criterion, optimizer,
                                      num_epochs=num_epochs)
        torch.save(model, f'{self.filename}/model.pth')

        self.createLossPlot()

        self.createAccPlot()

        return self.model

    def train_model(self, train_data_loader: DataLoader, test_data_loader: DataLoader,
                    model, criterion, optimizer, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            logging.debug(f'Epoch {epoch+1}/{num_epochs}')
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    data_loader = train_data_loader
                else:
                    model.eval()
                    data_loader = test_data_loader

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_corrects.double() / len(data_loader.dataset) * 100

                # Save statistics
                if phase == 'train':
                    self.training_loss.append(epoch_loss)
                    self.training_accuracy.append(epoch_acc.cpu())
                else:
                    self.validation_loss.append(epoch_loss)
                    self.validation_accuracy.append(epoch_acc.cpu())

                logging.debug(f'{phase} Loss: {epoch_loss:3.4f} Acc: {epoch_acc:3.4f}%')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        logging.debug(f'{self.training_loss}')
        logging.debug(f'{self.training_accuracy}')
        logging.debug(f'{self.validation_loss}')
        logging.debug(f'{self.validation_accuracy}')
        logging.debug(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60:2.0f}s')
        logging.debug(f'Best val Acc: {best_acc:3.4f}%')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def createLossPlot(self):
        plt.figure()
        plt.plot(self.training_loss, label='Zbiór treningowy')
        plt.plot(self.validation_loss, label='Zbiór walidacyjny')
        plt.title("Funkcja straty")
        plt.xlabel('Liczba epok')
        plt.ylabel('Wartość')
        plt.legend(frameon=False)
        plt.savefig(f'{self.filename}/loss.png')

    def createAccPlot(self):
        plt.figure()
        plt.plot(self.training_accuracy, label='Zbiór treningowy')
        plt.plot(self.validation_accuracy, label='Zbiór walidacyjny')
        plt.title("Dokładność")
        plt.xlabel('Liczba epok')
        plt.ylabel('Dokładność [%]')
        plt.legend(frameon=False)
        plt.savefig(f'{self.filename}/acc.png')
