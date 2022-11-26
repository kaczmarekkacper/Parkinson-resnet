import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal


def getPatientID(filename):
    return filename.split("_")[0]


def getMeasureNumber(filename):
    return int(filename.split("_")[1].split(".")[0])


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


def createWaveletPlot(x, cwtFunc, group, patientID, sensor):
    width = 100
    cwtmatr = cwtFunc()
    fig = plt.figure(frameon=False)
    # dpi = 1
    # fig.set_size_inches(224 / dpi, 224 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cwtmatr, extent=[0, x.values[-1], width, 1], cmap='PRGn', aspect='auto',
              vmax=cwtmatr.max(), vmin=cwtmatr.min())
    fig.savefig(f'plots/wavelets/{group}/{patientID}_{sensor}.jpg',
                format="jpg", bbox_inches="tight", pad_inches=0)  # , dpi=dpi)
    plt.close()
