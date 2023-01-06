import LearningUtils
import ModelTrainer
from torchvision import models
import logging
import sys

dataFolder = sys.argv[1]
path = f'/drive/My Drive/Parkinson/Data/{dataFolder}'
num_epochs = 250

wavelets = sys.argv[2:]

for wavelet in wavelets:
    print(f'Resnet for {wavelet}')
    imgdir = f"{path}/{wavelet}"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fhandler = logging.FileHandler(f'{imgdir}/log.txt', mode='w')
    consoleHandler = logging.StreamHandler(sys.stdout)
    logger.addHandler(fhandler)
    logger.addHandler(consoleHandler)

    classes = LearningUtils.get_classes(imgdir)
    train_data_loader, validation_data_loader, test_data_loader = LearningUtils.prepare_data(
        imgdir, 256)
    model_trainer = ModelTrainer.ModelTrainer(classes, imgdir)
    resnet = models.resnet18(pretrained=True)
    model = model_trainer.train_resnet(
        resnet, train_data_loader, validation_data_loader, num_epochs=num_epochs)
    model.eval()
    LearningUtils.predict_image(model, test_data_loader, classes)

    logger.removeHandler(fhandler)
    logger.removeHandler(consoleHandler)
