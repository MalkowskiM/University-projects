import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import customtkinter as ctk

import GaitCycle as gc


def myFilter(accData):
    """
    filtering accData
    :param accData: absolute acceleration data from accelerometer after getting rid of false readouts
    :return: filtered absolute acceleration
    """

    #accDataMedian = medfilt(file['Absolute acceleration (m/s^2)'], kernel_size=3)

    # Savitzky-Golay filter (low pass)
    window_size = 25
    dataSavgol = savgol_filter(accData, window_size, 2)

    #Butterworth filter (high pass)
    order = 5  # Rząd filtru
    cutoff_frequency = 0.1  # Częstotliwość odcięcia
    b, a = butter(order, cutoff_frequency, btype='high', analog=False, fs=200)
    dataButter = filtfilt(b, a, dataSavgol)

    #róźniczka
    #DataButter = lfilter([1,-1], 1, accDataButter)


    filteredData = dataButter
    return filteredData

def firstGraph(filepath, lowRange, highRange):
    """
    shows graph before and after filtration
    :param filepath: Str
    :param lowRange: int
    :param highRange: int
    """
    try:
        file = pd.read_csv(filepath)
    except Exception as error:
        print("There was an error loading the accData")

    # shortened by wrong results
    time = np.array(file['Time (s)'][lowRange:highRange])
    absAcceleration = np.array(file['Absolute acceleration (m/s^2)'][lowRange:highRange])

    filteredAcceleration = myFilter(absAcceleration)

    plt.figure(1, figsize=(12, 10), dpi=75)

    # creating the graph from unfiltered data
    plt.subplot(2, 1, 1, )
    plt.plot(time, absAcceleration)
    plt.xlim(left=4)
    plt.title("Accelerometer read before filtration")
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute acceleration (m/s^2)')

    # creating the graph from filtered accData
    plt.subplot(2, 1, 2)
    plt.plot(time, filteredAcceleration)
    # show peaks
    # plt.plot(time[peaks], filteredAcceleration[peaks], 'x', label='maksima lokalne', color='red')
    plt.xlim(left=4)
    plt.title("Accelerometer read after filtration")
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute acceleration (m/s^2)')
    plt.show()

def showGaitCycle(filepath, lowRange, highRange, distance):
    """
    shows graph with marked gait cycles and show mean swing and stance time means
    :param filepath: Str
    :param lowRange: int
    :param highRange: int
    :param distance: int
    """
    try:
        file = pd.read_csv(filepath)
    except Exception as error:
        print("There was an error loading the accData")

    # shortened by wrong results
    time = np.array(file['Time (s)'][lowRange:highRange])
    absAcceleration = np.array(file['Absolute acceleration (m/s^2)'][lowRange:highRange])

    # filtered accData
    filteredAcceleration = myFilter(absAcceleration)
    # local maxima
    peaks, _ = find_peaks(filteredAcceleration, distance=distance)

    plt.figure(figsize=(12, 4), dpi=150)

    sumSwingTime = 0
    sumStanceTime = 0
    nrOfCycles = 0
    for i in range(len(peaks)):
        if i + 1 != len(peaks):
            if i + 2 != len(peaks):
                if i % 2 == 0:
                    g = gc.GaitCycle(peaks[i], peaks[i + 1], peaks[i + 2],time)
                    g.gaitPlot(time, filteredAcceleration)
                    sumStanceTime += g.getStancePhaseTime()
                    sumSwingTime += g.getSwingPhaseTime()
                    nrOfCycles += 1
    meanStanceTime = sumStanceTime / nrOfCycles
    meanSwingTime = sumSwingTime / nrOfCycles
    plt.title(f"Gait cycles with distance between peaks = {distance} ")
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute acceleration (m/s^2)')
    plt.show()
    root = ctk.CTk()
    root.geometry("300x100")
    root.title("gait cycle stats")
    meanStanceLabel = ctk.CTkLabel(root, text=f"mean stance time = {meanStanceTime:.3f} s")
    meanStanceLabel.pack(pady = 10)
    meanSwingLabel = ctk.CTkLabel(root, text=f"mean swing time = {meanSwingTime:.3f} s")
    meanSwingLabel.pack(pady = 10)
    root.mainloop()


# # measured frequency
# print(1/(-file['Time (s)'][0] + file['Time (s)'][1]))  198.6
# print(7314/file.tail(1)['Time (s)'])  198.84
# frequency = 200





