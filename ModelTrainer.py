from __future__ import print_function, division

import copy
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


class ModelTrainer:

    def __init__(self, classes):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.training_loss = list()
        self.training_accuracy = list()
        self.validation_loss = list()
        self.validation_accuracy = list()
        self.model = None
        self.classes = classes

    def train_resnet(self, model, train_data_loader, test_data_loader, num_epochs=25):

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
        torch.save(model, 'resnet.pth')

        plt.plot(self.training_loss, label='Training loss')
        plt.plot(self.validation_loss, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()
        plt.ioff()
        plt.show()

        plt.savefig('plots/loss.png')

        plt.plot(self.training_accuracy, label='Training accuracy')
        plt.plot(self.validation_accuracy, label='Validation accuracy')
        plt.legend(frameon=False)
        plt.show()
        plt.ioff()
        plt.show()

        plt.savefig('plots/acc.png')

        return self.model

    def train_model(self, train_data_loader: DataLoader, test_data_loader: DataLoader,
                    model, criterion, optimizer, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    data_loader = train_data_loader
                else:
                    model.eval()  # Set model to evaluate mode
                    data_loader = test_data_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
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
                epoch_acc = running_corrects.double() / len(data_loader.dataset)

                # Save statistics
                if phase == 'train':
                    self.training_loss.append(epoch_loss)
                    self.training_accuracy.append(epoch_acc.cpu())
                else:
                    self.validation_loss.append(epoch_loss)
                    self.validation_accuracy.append(epoch_acc.cpu())

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
