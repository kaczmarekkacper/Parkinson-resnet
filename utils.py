import shutil
import pickle
import csv
import os
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def readData():
    col_names = ['SUM', 'FILENAME']
    data_file_colum_names = ['Time', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6',
                             'R7', 'R8', 'Force_Left', 'Force_Right']
    sums = pd.read_csv('Data/SHA256SUMS.txt', header=None,
                       sep=' ', names=col_names)
    all_filenames = sums['FILENAME']
    filenames = all_filenames[all_filenames.str.contains(r'\d.txt$')]
    records = {}
    for filename in filenames:
        patientID = getPatientID(filename)
        measureNumber = getMeasureNumber(filename)
        if patientID not in records:
            records[patientID] = {}
        records[patientID][f'data{measureNumber}'] = pd.read_csv(f'Data/{filename}', header=None, sep='\t',
                                                                 names=data_file_colum_names)
    return records


def getPatientID(filename):
    return filename.split("_")[0]


def getMeasureNumber(filename):
    return int(filename.split("_")[1].split(".")[0])


def createComparisonPlot(healthy_record, parkinson_record, sample_number):
    x_healthy_left = healthy_record.Time.head(sample_number)
    y_healthy_left = healthy_record.Force_Left.head(sample_number)
    x_healthy_right = healthy_record.Time.head(sample_number)
    y_healthy_right = healthy_record.Force_Right.head(sample_number)

    x_parkinson_left = parkinson_record.Time.head(sample_number)
    y_parkinson_left = parkinson_record.Force_Left.head(sample_number)
    x_parkinson_right = parkinson_record.Time.head(sample_number)
    y_parkinson_right = parkinson_record.Force_Right.head(sample_number)

    y_lim_value = pd.concat(
        [y_healthy_left, y_healthy_right, y_parkinson_left, y_parkinson_right]).max()

    fig, axs = plt.subplots(2, 2)
    healthy_plot_left = axs[0, 0]
    healthy_plot_left.set_title('Zdrowy')
    healthy_plot_left.plot(x_healthy_left, y_healthy_left)
    healthy_plot_left.set_ylim(0, y_lim_value)
    parkinson_plot_left = axs[0, 1]
    parkinson_plot_left.set_title('Parkinson')
    parkinson_plot_left.plot(x_parkinson_left, y_parkinson_left)
    parkinson_plot_left.set_ylim(0, y_lim_value)
    healthy_plot_right = axs[1, 0]
    healthy_plot_right.plot(x_healthy_right, y_healthy_right)
    healthy_plot_right.set_ylim(0, y_lim_value)
    parkinson_plot_right = axs[1, 1]
    parkinson_plot_right.plot(x_parkinson_right, y_parkinson_right)
    parkinson_plot_right.set_ylim(0, y_lim_value)
    fig.text(0.5, 0.04, 'Czas [s]', ha='center')
    fig.text(0.02, 0.5, 'Siła nacisku [N]', va='center', rotation='vertical')

    plt.savefig("plots/gait-comparison.pdf", format="pdf", bbox_inches="tight")


def createAllSensorPlot(patient_record, filename, sample_number=1000):
    x_l1, y_l1 = getXYFromPatient(
        patient_record, lambda x: x.L1, sample_number)
    x_l2, y_l2 = getXYFromPatient(
        patient_record, lambda x: x.L2, sample_number)
    x_l3, y_l3 = getXYFromPatient(
        patient_record, lambda x: x.L3, sample_number)
    x_l4, y_l4 = getXYFromPatient(
        patient_record, lambda x: x.L4, sample_number)
    x_l5, y_l5 = getXYFromPatient(
        patient_record, lambda x: x.L5, sample_number)
    x_l6, y_l6 = getXYFromPatient(
        patient_record, lambda x: x.L6, sample_number)
    x_l7, y_l7 = getXYFromPatient(
        patient_record, lambda x: x.L7, sample_number)
    x_l8, y_l8 = getXYFromPatient(
        patient_record, lambda x: x.L8, sample_number)

    x_r1, y_r1 = getXYFromPatient(
        patient_record, lambda x: x.R1, sample_number)
    x_r2, y_r2 = getXYFromPatient(
        patient_record, lambda x: x.R2, sample_number)
    x_r3, y_r3 = getXYFromPatient(
        patient_record, lambda x: x.R3, sample_number)
    x_r4, y_r4 = getXYFromPatient(
        patient_record, lambda x: x.R4, sample_number)
    x_r5, y_r5 = getXYFromPatient(
        patient_record, lambda x: x.R5, sample_number)
    x_r6, y_r6 = getXYFromPatient(
        patient_record, lambda x: x.R6, sample_number)
    x_r7, y_r7 = getXYFromPatient(
        patient_record, lambda x: x.R7, sample_number)
    x_r8, y_r8 = getXYFromPatient(
        patient_record, lambda x: x.R8, sample_number)

    y_lim_value = pd.concat([y_l1, y_l2, y_l3, y_l4, y_l5, y_l6, y_l7, y_l8,
                             y_r1, y_r2, y_r3, y_r4, y_r5, y_r6, y_r7, y_r8]).max()

    fig, axs = plt.subplots(8, 2)
    plot = axs[0, 0]
    plot.set_title('Lewa noga')
    plot.plot(x_l1, y_l1)
    plot.set_ylim(0, y_lim_value)
    plot = axs[1, 0]
    plot.plot(x_l2, y_l2)
    plot.set_ylim(0, y_lim_value)
    plot = axs[2, 0]
    plot.plot(x_l3, y_l3)
    plot.set_ylim(0, y_lim_value)
    plot = axs[3, 0]
    plot.plot(x_l4, y_l4)
    plot.set_ylim(0, y_lim_value)
    plot = axs[4, 0]
    plot.plot(x_l5, y_l5)
    plot.set_ylim(0, y_lim_value)
    plot = axs[5, 0]
    plot.plot(x_l6, y_l6)
    plot.set_ylim(0, y_lim_value)
    plot = axs[6, 0]
    plot.plot(x_l7, y_l7)
    plot.set_ylim(0, y_lim_value)
    plot = axs[7, 0]
    plot.plot(x_l8, y_l8)
    plot.set_ylim(0, y_lim_value)
    plot = axs[0, 1]
    plot.set_title('Prawa noga')
    plot.plot(x_r1, y_r1)
    plot.set_ylim(0, y_lim_value)
    plot = axs[1, 1]
    plot.plot(x_r2, y_r2)
    plot.set_ylim(0, y_lim_value)
    plot = axs[2, 1]
    plot.plot(x_r3, y_r3)
    plot.set_ylim(0, y_lim_value)
    plot = axs[3, 1]
    plot.plot(x_r4, y_r4)
    plot.set_ylim(0, y_lim_value)
    plot = axs[4, 1]
    plot.plot(x_r5, y_r5)
    plot.set_ylim(0, y_lim_value)
    plot = axs[5, 1]
    plot.plot(x_r6, y_r6)
    plot.set_ylim(0, y_lim_value)
    plot = axs[6, 1]
    plot.plot(x_r7, y_r7)
    plot.set_ylim(0, y_lim_value)
    plot = axs[7, 1]
    plot.plot(x_r8, y_r8)
    plot.set_ylim(0, y_lim_value)

    plt.savefig(f'plots/{filename}.pdf', format="pdf", bbox_inches="tight")


def getXYFromPatient(patient_record, getField, sample_number=1000):
    x = patient_record.Time.head(sample_number)
    y = getField(patient_record).head(sample_number)
    return x, y


def createWaveletPlot(x, cwtFunc, widths, path, format):
    cwtmatr = cwtFunc()
    plt.figure()
    # dpi = 1
    # fig.set_size_inches(224 / dpi, 224 / dpi)
    plt.imshow(cwtmatr, extent=[0, x.values[-1], widths[-1], widths[0]], cmap='PRGn',
               aspect='auto', vmax=cwtmatr.max(), vmin=cwtmatr.min())
    plt.savefig(f'{path}.{format}',
                format=format, bbox_inches="tight", pad_inches=0)  # , dpi=dpi)


def createWaveletPlotForResnet(x, cwtFunc, widths, path, format):
    cwtmatr = cwtFunc()
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cwtmatr, extent=[0, x.values[-1], widths[-1], widths[0]], cmap='PRGn',
              aspect='auto', vmax=cwtmatr.max(), vmin=cwtmatr.min())
    fig.savefig(f'{path}.{format}',
                format=format, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def createWaveletPlotForResnet224x224(x, cwtFunc, widths, path, format):
    cwtmatr = cwtFunc()
    fig = plt.figure(frameon=False)
    dpi = 1
    fig.set_size_inches(224 / dpi, 224 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cwtmatr, extent=[0, x.values[-1], widths[-1], widths[0]], cmap='PRGn',
              aspect='auto', vmax=cwtmatr.max(), vmin=cwtmatr.min())
    fig.savefig(f'{path}.{format}',
                format=format, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


def createWaveletPlotForResnet640x224(x, cwtFunc, widths, path, format):
    cwtmatr = cwtFunc()
    fig = plt.figure(frameon=False)
    dpi = 1
    fig.set_size_inches(640 / dpi, 224 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cwtmatr, extent=[0, x.values[-1], widths[-1], widths[0]], cmap='PRGn',
              aspect='auto', vmax=cwtmatr.max(), vmin=cwtmatr.min())
    fig.savefig(f'{path}.{format}',
                format=format, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


def getParkinsonStatus(patients, id):
    hoehnYahr = patients[patients['ID'] == id]['HoehnYahr'].values[0]
    parkinson = 'Parkinson' if hoehnYahr > 0 else 'Healthy'
    return parkinson


def getHoehnYahrStatus(patients, id):
    hoehnYahr = patients[patients['ID'] == id]['HoehnYahr'].values[0]
    hoehnYahr = hoehnYahr if hoehnYahr > 0 else 0.0
    return hoehnYahr


def createDatabase(record, id, sensors, splits_list, howManyInSample, main_path, parkinson, widths, w):
    for sensor in sensors:
        for i in range(1, len(splits_list)):
            for wavelet in ['Ricker', 'Morlet']:
                func, name = sensor
                start = splits_list[i-1]
                end = splits_list[i]
                if end - start == howManyInSample:
                    x = record.Time[start:end]
                    y = func(record).abs()[start:end]
                    path = f'{main_path}/{wavelet}/{parkinson}/{id}_{name}_{i}'
                    def cwtFunc(): return signal.cwt(y, signal.ricker, widths, dtype='float64')
                    if wavelet == 'Morlet':
                        def cwtFunc(): return signal.cwt(
                            y, signal.morlet2, widths, dtype='float64', w=w)
                    createWaveletPlotForResnet640x224(
                        x, cwtFunc, widths, path, 'jpg')


def removeDatasetFolders(path):
    shutil.rmtree(f'{path}', ignore_errors=True)


def createDatasetFolders(main_path):
    os.makedirs(f'{main_path}/Healthy', exist_ok=True)
    os.makedirs(f'{main_path}/Parkinson', exist_ok=True)


def datasetExists(main_path):
    return os.path.exists(main_path)


def createDatasets(main_path):
    directory = f"{main_path}"
    output = f"{main_path}"
    split = "70/20/10"

    train_f = open(f"{output}/train", 'w', encoding='UTF8', newline='')
    train_writer = csv.writer(train_f)
    test_f = open(f"{output}/test", 'w', encoding='UTF8', newline='')
    test_writer = csv.writer(test_f)
    validation_f = open(f"{output}/validation", 'w',
                        encoding='UTF8', newline='')
    validation_writer = csv.writer(validation_f)

    s = split.split('/')
    splits = {
        "train": int(s[0]),
        "validation": int(s[1]),
        "test": int(s[2])
    }

    classes = {}
    for idx, c in enumerate(os.listdir(directory)):
        if not os.path.isdir(os.path.join(directory, c)):
            continue
        classes[idx] = c

        images = os.listdir(f"{directory}/{c}")
        class_count = len(images)
        for imidx, filename in enumerate(images):
            row = [f"{c}/{filename}", idx]
            if imidx < class_count * splits["train"] / 100:
                train_writer.writerow(row)
            elif imidx < class_count * (splits["train"] + splits["test"]) / 100:
                test_writer.writerow(row)
            else:
                validation_writer.writerow(row)

    with open(f"{output}/utils", 'wb') as f:
        pickle.dump(classes, f)

    train_f.close()
    test_f.close()
    validation_f.close()
