import customtkinter as ctk

import GUILogic
class GUI:

    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("200x50")
        self.root.title("Projekt2_ISMED")

        self.button = ctk.CTkButton(self.root, text="chose data", command=GUILogic.choseData)
        self.button.pack(pady =10)

        self.root.mainloop()



if __name__ == '__main__':
    GUI();