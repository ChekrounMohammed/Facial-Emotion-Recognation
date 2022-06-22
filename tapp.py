from re import L
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
<<<<<<< HEAD
model = load_model('modelfin.h5')
ups=0
md=0


=======
ups=0
md=0
lab=None
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e
def Emotion():
    global filepath
    global panelD
    global md
    global lab
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
    temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)   
    img =Image.fromarray(temp)
    photo = ImageTk.PhotoImage(img)   
<<<<<<< HEAD
    lab.configure(image=photo,pady=10)
    lab.image = photo
    #panelD.add(lab)
=======
    if md==0:
        panelD.remove(lab)
        lab = Label(panelD,image=photo,pady=10)    
        lab.image = photo   
        panelD.add(lab)
        md=md+1
    #lab.place(x=300,y=100)
    else :
        #panelD.remove(lab)
        lab.configure(image=photo,pady=10)
        lab.image = photo
        #panelD.add(lab)
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e



def UploadAction():
    global filepath
    global panelD
    global ups
    global lab
<<<<<<< HEAD
    global exit
    global btngn
    exit = True
=======
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e
    filepath = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    print('Selected:', filepath)
    monimage = Image.open(filepath)
    monimage = monimage.resize((500, 500), Image.LANCZOS)
    photo = ImageTk.PhotoImage(monimage)   ## Création d'une image compatible Tkinter
    if ups==0 :
<<<<<<< HEAD
        
        btngn = Button(panelD, text ="get emotion" ,width=20 ,height=5, bg='#567', fg='White' ,command=Emotion)
        btngn.pack(side=LEFT)
        #panelD.add(btngn)
        lab.configure(image=photo,pady=10)
        lab.image = photo
=======
        btngn = Button(panelD, text ="get emotion" ,width=10 ,height=3, bg='#567', fg='White' ,command=Emotion)
        panelD.add(btngn)
        lab = Label(panelD,image=photo,pady=10)    
        lab.image = photo   
        panelD.add(lab)
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e
        #lab.place(x=300,y=100)
        
        ups=ups+1
    else :
<<<<<<< HEAD
        
=======
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e
        #panelD.remove(lab)
        lab.configure(image=photo,pady=10)
        lab.image = photo
        #panelD.add(lab)
<<<<<<< HEAD
labv=None
cap =None
def realtime():
    global panelD
    global lab
    global cap
    global exit 
    global btngn
    panelD.forget(btngn)
    exit = False
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap= cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError('Cannot open camera')
      
    video_stream()
    

def video_stream():
    global lab
    global model
    global cap
    global exit
    state = {0 : "Neutral",1 : "Happy",2 : "Sad",3 : "Surprise",4 : "anger",5 : "fear",6 : "disgust"}
    _,frame= cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, confidence = cv.detect_face(frame)
    if faces :
        for f in faces:
            (startX, startY)=f[0], f[1]
            (endX, endY)= f[2], f[3]
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
            
            face_crop = np.copy(img[startY:endY,startX:endX])
            try :
                face_crop = cv2.resize(face_crop, (48,48))
                face_crop = np.array(face_crop)
                #face_crop = face_crop[0:48,0:48,2:]
                face_crop = np.expand_dims(face_crop, 0)
                cf = model.predict(face_crop)
            
                score = np.argmax(cf)
                label = state[score]
                idx = 100 * np.max(cf)
                label = label +" pr:"+ str(int(idx))+"%"
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            except:
                print('empty*****')

    temp = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)   
    img =Image.fromarray(temp)
    photo = ImageTk.PhotoImage(img)   
    
    lab.configure(image=photo,pady=10)
    lab.image = photo
    if not exit:
        lab.after(1,video_stream)
    #panelD.add(lab)
       
=======

    
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e



filepath = None



root = tk.Tk()
root.geometry("1000x800")
panelA = None

panelA = PanedWindow(bd=4,relief='raised')
panelA.pack(fill=BOTH, expand=1)


panelB =PanedWindow(panelA,orient=VERTICAL,bd=4)

panelA.add(panelB)

panelC =PanedWindow(panelB,orient=HORIZONTAL,bd=4)
panelB.add(panelC)


button1 = tk.Button(panelC, text='Upload an image',bg='#567', fg='White',width=20 ,height=3, command=UploadAction)
button1.pack()

<<<<<<< HEAD
button2 = tk.Button(panelC, text='Video stream',bg='#567', width=20 ,height=3,fg='White',command= realtime )
=======
button2 = tk.Button(panelC, text='Video stream',bg='#567', width=20 ,height=3,fg='White' )
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e
button2.pack()

panelC.add(button1)
panelC.add(button2)

panelD = PanedWindow(panelB,orient=HORIZONTAL,bd=4)
panelB.add(panelD)
<<<<<<< HEAD

btngn =Button()
panelD.add(btngn)
lab =Label(panelD)
lab.pack(side=RIGHT)
panelD.add(lab)
=======
>>>>>>> e7751ddadbe6654a1b3266c1dabaaa97139e5e4e

root.mainloop()



