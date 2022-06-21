import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog

class Layout:
    
    def __init__(self,master):
        self.master = master
        self.rootgeometry()

        self.button = Button(self.master, text='Upload', command=self.loadbackground)
        self.button.pack()
        
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(fill=BOTH, expand=True)

        self.background_image = None
        self.image_copy = None
        self.background = None

        self.label = tk.Label(self.canvas)
        self.label.pack(fill='both', expand=True)
        

    def loadbackground(self):

        self.background_image = Image.open(self.openfn())
        self.image_copy = self.background_image.copy()
        
        self.background = ImageTk.PhotoImage(self.background_image.resize((self.canvas.winfo_width(), self.canvas.winfo_height())))
        self.label.configure(image=self.background)
        self.label.bind('<Configure>',self.resizeimage)

    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename
    

    def rootgeometry(self):
        x=int(self.master.winfo_screenwidth()*0.7)
        y=int(self.master.winfo_screenheight()*0.7)
        z = str(x) +'x'+str(y)
        self.master.geometry(z)

    def resizeimage(self,event):
        image = self.image_copy.resize((event.width, event.height))
        self.image1 = ImageTk.PhotoImage(image)
        self.label.config(image = self.image1)
        

root = tk.Tk()
a = Layout(root)
root.mainloop()