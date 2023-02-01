import re
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def readLogFile(dataset):
    filename = f'./Datasets/Binary/{dataset}/log.txt'
    content = ""
    with open(filename, "r") as f:
        content = f.read()
    return content


datasets = ['AllSensorsDiff-224x224-5sec', 'AllSensorsDiff-640x224-10sec', 'AllSensorsDiff-640x224-10sec-1-20', 'AllSensorsDiff-640x224-10sec-1-20-Adam',
            'AllSensorsDiff-640x224-10sec-1-100', 'AllSensorsDiff-640x224-10sec-20-49', 'AllSensorsDiff-640x480-5sec', 'AllSensorsDiff-640x480-10sec']

for dataset in datasets:
    logFiles = []
    for wavelet in ['Ricker', 'Morlet']:
        logFiles.append(readLogFile(f'{dataset}/{wavelet}'))
    accuracy = []
    loss = []
    for logFile in logFiles:
        res = re.search(r'\[[0-9]*.', logFile)
        start = res.start()
        end = logFile.find('Training complete in')
        arrays = logFile[start:end].split('\n')
        def convertToFloat(l): return [float(x) for x in l]

        def createListFromString(x): return convertToFloat(
            x.strip('][').split(', '))

        loss.append([createListFromString(arrays[0]),
                     createListFromString(arrays[2])])

        def trimTensor(x): return x.replace(
            'tensor(', '').replace(', dtype=torch.float64)', '')
        arrays[1] = trimTensor(arrays[1])
        arrays[3] = trimTensor(arrays[3])
        accuracy.append([createListFromString(arrays[1]),
                         createListFromString(arrays[3])])

    rickerAcc = accuracy[0]
    morletAcc = accuracy[1]
    rickerLoss = loss[0]
    morletLoss = loss[1]

    def flatList(x): return [item for sublist in x for item in sublist]

    allAcc = flatList([*rickerAcc, *morletAcc])
    allLoss = flatList([*rickerLoss, *morletLoss])
    allAccMin = math.floor(min(allAcc))
    allAccMax = math.ceil(max(allAcc))
    allLossMin = math.floor(min(allLoss)*100)/100
    allLossMax = math.ceil(max(allLoss)*100)/100

    fig, axs = plt.subplots(2, 2)
    morletAccFig = axs[0, 0]
    morletAccFig.set_title('Falka Morleta')
    morletAccFig.plot(morletAcc[0], label='Zbiór treningowy')
    morletAccFig.plot(morletAcc[1], label='Zbiór walidacyjny')
    # morletAccFig.set_ylim(allAccMin, allAccMax)
    morletAccFig.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    morletAccFig.yaxis.set_ticks(
        np.linspace(allAccMin, allAccMax, 5))
    rickerAccFig = axs[0, 1]
    rickerAccFig.set_title('Falka Rickera')
    rickerAccFig.plot(rickerAcc[0], label='Zbiór treningowy')
    rickerAccFig.plot(rickerAcc[1], label='Zbiór walidacyjny')
    # rickerAccFig.set_ylim(allAccMin, allAccMax)
    rickerAccFig.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    rickerAccFig.yaxis.set_ticks(
        np.linspace(allAccMin, allAccMax, 5))
    rickerLossFig = axs[1, 1]
    rickerLossFig.plot(rickerLoss[0], label='Zbiór treningowy')
    rickerLossFig.plot(rickerLoss[1], label='Zbiór walidacyjny')
    # rickerLossFig.set_ylim(allLossMin, allLossMax)
    rickerLossFig.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    rickerLossFig.yaxis.set_ticks(
        np.linspace(allLossMin, allLossMax, 5))
    morletLossFig = axs[1, 0]
    morletLossFig.plot(morletLoss[0], label='Zbiór treningowy')
    morletLossFig.plot(morletLoss[1], label='Zbiór walidacyjny')
    # morletLossFig.set_ylim(allLossMin, allLossMax)
    morletLossFig.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    morletLossFig.yaxis.set_ticks(
        np.linspace(allLossMin, allLossMax, 5))
    fig.text(0.5, 0.04, 'Liczba epok', ha='center')
    fig.text(0.02, 0.3, 'Funkcja straty', va='center', rotation='vertical')
    fig.text(0.02, 0.7, 'Dokładność (%)', va='center', rotation='vertical')
    # plt.legend(frameon=False)
    # plt.show()

    plt.savefig(f'./Datasets/Binary/{dataset}/plot.pdf',
                format="pdf", bbox_inches="tight")
