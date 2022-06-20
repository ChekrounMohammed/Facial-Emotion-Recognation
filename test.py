import cv2
import numpy as np
import matplotlib.pyplot as plt

state = {0 : "Neutral",1 : "Happy",2 : "Sad",3 : "Surprise",4 : "anger",5 : "fear",6 : "disgust"}

from tensorflow import keras
model = keras.models.load_model('modelfin.h5')




path = "haarcascade_frontalface_default.xml"
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

face_roi = np.zeros((500,500))
while True:
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0))
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex,ey, ew,eh) in facess:
                face_roi = roi_gray[ey: ey+eh, ex:ex+ew]
                
    final_img = cv2.resize(face_roi, (48,48))
    final_img = np.expand_dims(final_img, axis = 0)
    final_img = final_img /255.
    
    
    prediction = model.predict(final_img.reshape((1,48,48,1)))
    status = state[prediction[0]]
    x1,y1,w1,h1 = 0,0,175,75
    cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,0,0), -1)
    cv2.putText(frame, status, (x1+int(w1/10), y1+ int(h1/2)), font, 0.7, (0,0,255), 2)
    cv2.putText(frame, status, (100,150), font, 3, (0,0,255), 2, cv2.LINE_4)
    #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
    
    
    cv2.imshow('face Emotion Reco', frame)
    
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()