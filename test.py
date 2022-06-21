import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv


from tensorflow import keras
model = keras.models.load_model('modelfin.h5')

state = {0 : "Neutral",1 : "Happy",2 : "Sad",3 : "Surprise",4 : "anger",5 : "fear",6 : "disgust"}


#path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

rectangle_bgr = (255,255,255)
img = np.zeros((500,500))

text = "some text in box"

(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness = 1)[0]

text_offset_x = 10
text_offset_y = img.shape[0]-25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x+text_width +2, text_offset_y - text_height - 2 )  )
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale = font_scale, color=(0,0,0) , thickness = 1 )

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cannot open camera')

#face_roi = np.zeros((500,500))
while True:
    ret, frame = cap.read()
    #faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
        
    cv2.imshow('face Emotion Reco', frame)
    
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()