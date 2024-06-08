import matplotlib.pyplot as plt
import numpy as np

class GaitCycle:
    """
    Class representing one gait cycle
    """
    # constructor
    def __init__(self, stancePhaseStart, swingPhaseStart, nextStep, time):
        self.__stancePhaseStart = stancePhaseStart
        self.__swingPhaseStart = swingPhaseStart
        self.__nextStep = nextStep
        self.__stancePhaseStartTime = time[stancePhaseStart]
        self.__swingPhaseStartTime = time[swingPhaseStart]
        self.__nextStepTime = time[nextStep]
        self.__stancePhaseTime = abs(time[stancePhaseStart] - time[swingPhaseStart])
        self.__swingPhaseTime = abs(time[swingPhaseStart] - time[nextStep])

    # getters
    def getStancePhaseStart(self):
        return self.__stancePhaseStart
    def getSwingPhaseStart(self):
        return self.__swingPhaseStart
    def getNextStep(self):
        return self.__nextStep
    def getStancePhaseStartTime(self):
        return self.__stancePhaseStartTime
    def getSwingPhaseStartTime(self):
        return self.__swingPhaseStartTime
    def getNextStepTime(self):
        return self.__nextStepTime
    def getStancePhaseTime(self):
        return self.__stancePhaseTime
    def getSwingPhaseTime(self):
        return self.__swingPhaseTime

    def gaitPlot(self, x, y):
        """
        plots one gait cycle
        :param x: time data
        :param y: filtered absolute acceleration data
        """
        plt.plot(x[self.__stancePhaseStart: self.__swingPhaseStart + 1], y[self.__stancePhaseStart: self.__swingPhaseStart + 1], color='red', label='stance phase')
        plt.plot(x[self.__swingPhaseStart: self.__nextStep + 1], y[self.__swingPhaseStart: self.__nextStep + 1], color='blue', label='swing phase')

        # plt.axvline(x = self.__stancePhaseStartTime)
        # plt.axvline(x = self.__swingPhaseStartTime)
        plt.plot(x[self.__stancePhaseStart: self.__swingPhaseStart + 1], -5.5 * np.ones([len(x[self.__stancePhaseStart: self.__swingPhaseStart + 1]), 1]), color = 'green', label = 'stance phase time', linewidth = 5)
        plt.plot(x[self.__swingPhaseStart: self.__nextStep + 1], -5.5 * np.ones([len(x[self.__swingPhaseStart: self.__nextStep + 1]), 1]), color = 'yellow', label = 'swing phase time', linewidth = 5)
        # plt.plot(x[self.__stancePhaseStart], -1, 'o', color='black', markersize = 4)
        plt.text(x[self.__stancePhaseStart + 50], -5.4,f"{self.__stancePhaseTime:.2f}s")
        plt.text(x[self.__swingPhaseStart + 50], -5.4,f"{self.__swingPhaseTime:.2f}s")
        # plt.plot(x[self.__swingPhaseStart], -1, '.', color='black')



