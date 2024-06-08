import customtkinter as ctk

import AccelerometerRead
from ShowDataGUI import ShowDataGUI


class ChooseDataGUI:

    def __init__(self, filepath, size):
        self.root = ctk.CTk()
        self.root.geometry(("500x200"))
        self.root.title("Chosing data")

        self.sliderFrame = ctk.CTkFrame(self.root)
        self.sliderFrame.columnconfigure(0,weight =1)
        self.sliderFrame.columnconfigure(1,weight =1)

        def slidingLow(value):
            self.lowSliderLabel.configure(text = int(value))

        def slidingHigh(value):
            self.highSliderLabel.configure(text = int(value))

        self.lowSlider = ctk.CTkSlider(
            self.sliderFrame,
            from_ = 0,
            to = size/2-25,
            command= slidingLow,
        )
        self.lowSlider.grid(row = 0, column = 0)
        self.lowSlider.set(0)
        self.lowSliderLabel = ctk.CTkLabel(self.sliderFrame, text = self.lowSlider.get())
        self.lowSliderLabel.grid(row = 1, column = 0)
        self.highSlider = ctk.CTkSlider(
            self.sliderFrame,
            from_ = size/2,
            to = size,
            command= slidingHigh,
        )
        self.highSlider.grid(row = 0, column = 1)
        self.highSlider.set(size)
        self.highSliderLabel = ctk.CTkLabel(self.sliderFrame, text= self.highSlider.get())
        self.highSliderLabel.grid(row=1, column=1)

        self.sliderFrame.pack(pady = 10)


        self.showDataButton = ctk.CTkButton(
            self.root,
            text = "show data",
            command = lambda :AccelerometerRead.firstGraph(filepath, int(self.lowSlider.get()), int(self.highSlider.get())))
        self.showDataButton.pack(pady = 10)
        self.confirmButton = ctk.CTkButton(
            self.root,
            text = "confirm",
            command = lambda :ShowDataGUI(filepath, int(self.lowSlider.get()), int(self.highSlider.get()))
        )
        self.confirmButton.pack(pady = 10)

        self.root.mainloop()


