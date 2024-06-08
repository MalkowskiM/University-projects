import customtkinter as ctk

import AccelerometerRead

class ShowDataGUI:

    def __init__(self,filepath , lowRange, highRange):
        self.root = ctk.CTk()
        self.root.geometry(("200x100"))
        self.root.title("Gait cycle")

        self.entry = ctk.CTkEntry(self.root, placeholder_text= "distance between peaks")
        self.entry.pack(pady = 10)
        print(self.entry.get())


        self.saveButton = ctk.CTkButton(
            self.root,
            text = "confirm",
            command= lambda: AccelerometerRead.showGaitCycle(filepath, lowRange, highRange, int(self.entry.get()))
        )
        self.saveButton.pack(pady = 10)
        self.root.mainloop()
