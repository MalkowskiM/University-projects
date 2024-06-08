from customtkinter import filedialog as fd
import customtkinter as ctk
import numpy as np
import pandas as pd

import AccelerometerRead
from ChooseDataGUI import ChooseDataGUI

def choseData():
    """
    function used to initialize the csv file and ChooseDataGUI
    """
    filepath = fd.askopenfilename(filetypes=(('csv files', '*.csv'), ('All files', '*.*')), title="chose your file")
    try:
        file = pd.read_csv(filepath)
    except Exception as error:
        print("There was an error loading the accData")

    ChooseDataGUI(filepath, len(file))









