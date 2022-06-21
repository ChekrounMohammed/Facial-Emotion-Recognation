import tkinter as tk
from tkinter import *
from PIL import Image
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import cv2
import cvlib as cv
from tkinter import filedialog
from tensorflow.keras.models import load_model

def Emotion():
    global filepath
    global panelA
    #global window
    labels = {0 : "Neutral",1 : "Happy",2 : "Sad",3 : "Surprise",4 : "anger",5 : "fear",6 : "disgust"}
    model = load_model("modelfin.h5")
    image_path = filepath
    img = cv2.imread( image_path)
    img = cv2.resize(img, (512, 512))
    temp = img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face, confidence = cv.detect_face(temp)
    for  f in face :
        (startX, startY)=f[0], f[1]
        (endX, endY)= f[2], f[3]
        if (f[2]>512 or f[3]>512):
            continue
        cv2.rectangle(temp, (startX,startY), (endX,endY), (0,255,0), 2)
        try :
            face_crop = np.copy(img[startY:endY,startX:endX])
            face_crop = cv2.resize(face_crop, (48,48))
            face_crop = np.array(face_crop)
            #face_crop = np.stack(face_crop, axis=0)
            face_crop = np.expand_dims(face_crop, 0)
        except Exception as e:
            print(str(e))
        cf = model.predict(face_crop)
        score = np.argmax(cf)
        label = labels[score]
        idx = 100 * np.max(cf)
        label = label +" pr:"+ str(int(idx))+"%"
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(temp, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    img =Image.fromarray(temp)
    photo = ImageTk.PhotoImage(img)   
    if panelA is None:
        panelA = Label(image=photo)    
        panelA.image = photo            
        panelA.pack(side="center", padx=10,pady=10)
    else :
        panelA.configure(image=photo)
        panelA.image = photo	



def UploadAction(event=None):
    global filepath
    global panelA

    filepath = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('Selected:', filepath)
    monimage = Image.open(filepath)
    monimage = monimage.resize((500, 500), Image.LANCZOS)
    photo = ImageTk.PhotoImage(monimage)   ## Création d'une image compatible Tkinter
    if panelA is None:
        panelA = Label(image=photo,pady=10)    
        panelA.image = photo   

        panelA.place(x=300,y=100)
    else :
        panelA.configure(image=photo,pady=10)
        panelA.image = photo
        panelA.place(x=300,y=100)	

    btngn = Button(root, text ="get emotion" ,width=10 , bg='#567', fg='White' ,command=Emotion)
    btngn.place(x=550, y=50)

def resize_image(event):
    new_width = event.width
    new_height = event.height
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    label.config(image = photo)
    label.image = photo #avoid garbage collection

filepath = None



root = tk.Tk()
root.geometry("1000x800")
panelA = None
button = tk.Button(root, text='Upload an image',bg='#567', fg='White' , command=UploadAction)
button.pack()



img = Image.open("backgroundd.jpg")
copy_of_image = img.copy()
photo = ImageTk.PhotoImage(img)
label = Label(root, image = photo)
label.bind('<Configure>', resize_image)
label.pack(fill=BOTH , expand = YES)
root.mainloop()



