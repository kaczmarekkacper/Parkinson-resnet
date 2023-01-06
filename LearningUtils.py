import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
import importlib
import logging
import sys

import PatientDataset
importlib.reload(PatientDataset)


def get_classes(imgdir):
    with open(f"{imgdir}/utils", 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_data(imgdir, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    training_data = PatientDataset.PatientDataset(
        annotations_file=f"{imgdir}/train",
        img_dir=imgdir,
        transform=train_transform
    )
    test_data = PatientDataset.PatientDataset(
        annotations_file=f"{imgdir}/test",
        img_dir=imgdir,
        transform=test_transform
    )
    validation_data = PatientDataset.PatientDataset(
        annotations_file=f"{imgdir}/validation",
        img_dir=imgdir,
        transform=test_transform
    )

    train_data_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, validation_data_loader, test_data_loader


def predict_image(model, data_loader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confusion_matrix = torch.zeros(len(classes), len(classes))
    corrects = 0.0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    acc = corrects.double() / len(data_loader.dataset)
    logging.debug(f'Test dataset Accuracy: {acc.cpu().detach().numpy()}')
    logging.debug(
        f'Test dataset Confusion_matrix \n{confusion_matrix.cpu().detach().numpy()}')
