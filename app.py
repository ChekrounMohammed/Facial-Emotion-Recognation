import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import cvlib as cv
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,RTCConfiguration ,WebRtcMode
import av
import threading 



if __name__ == '__main__':

    RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.set_page_config(page_title="Facial Emotion Detection", page_icon="🤖")


    st.title('Emotion Detection from images')
    st.subheader('Chekroun & Ramadi ')

    task_list = ['Image','Video Stream']
    with st.sidebar:
        st.title('Task Selection')
        task_name=st.selectbox("Select your task :", task_list)
    st.title(task_name)

    if task_name == task_list[0]:

        file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if file is not None:

            image = Image.open(file)
            try:
                image = image.save("img.jpg")
                image = cv2.imread("img.jpg")
            except:
                image = image.save("img.png")
                image = cv2.imread("img.png")               
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            labels = {0 : "Neutral",1 : "Happy",2 : "Sad",3 : "Surprise",4 : "anger",5 : "fear",6 : "disgust"}
            img = cv2.resize(img, (512, 512))
            #img = np.stack(img, axis=0)
            img = np.repeat(img[..., np.newaxis], 3, -1)
            img = img.reshape((512,512,3))
            model = load_model('modelfin.h5')
            #img = cv2.imread(image_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face, confidence = cv.detect_face(img)
            for  f in face :
                (startX, startY)=f[0], f[1]
                (endX, endY)= f[2], f[3]
                if (f[2] > 512 or f[3] > 512):
                    break
                cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
                
                face_crop = np.copy(img[startY:endY,startX:endX])
                
                face_crop = cv2.resize(face_crop, (48,48))
                face_crop = np.array(face_crop)
                face_crop = face_crop[0:48,0:48,2:]
                face_crop = np.expand_dims(face_crop, 0)
                
                cf = model.predict(face_crop)
                
                score = np.argmax(cf)
                label = labels[score]
                idx = 100 * np.max(cf)
                label = label +" pr:"+ str(int(idx))+"%"
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            st.image(
                        img,
                        caption=f"Image detected",
                        use_column_width=True,
                    )
    if task_name == task_list[1]:
        style_list = ['color', 'black and white']

        st.sidebar.header('Style Selection')
        style_selection = st.sidebar.selectbox("Choose your style:", style_list)

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.model_lock = threading.Lock()
                self.style = style_list[0]

            def update_style(self, new_style):
                if self.style != new_style:
                    with self.model_lock:
                        self.style = new_style

            def recv(self, frame):
                # img = frame.to_ndarray(format="bgr24")
                img = frame.to_ndarray(format="bgr24")
                temp=img
                if self.style == style_list[1]:
                    img = img.convert("L")
                h,w,c =img.shape
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                labels = {0 : "Neutral",1 : "Happy",2 : "Sad",3 : "Surprise",4 : "anger",5 : "fear",6 : "disgust"}
                #img = cv2.resize(img, (512, 512))
                #img = np.stack(img, axis=0)
                img = np.repeat(img[..., np.newaxis], 3, -1)
                
                img = img.reshape((h,w,3))
                model = load_model('modelfin.h5')
                #img = cv2.imread(image_path)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face, confidence = cv.detect_face(img)
                for  f in face :
                    (startX, startY)=f[0], f[1]
                    (endX, endY)= f[2], f[3]
                    cv2.rectangle(temp, (startX,startY), (endX,endY), (0,255,0), 2)
                    
                    face_crop = np.copy(img[startY:endY,startX:endX])
                    
                    face_crop = cv2.resize(face_crop, (48,48))
                    face_crop = np.array(face_crop)
                    face_crop = face_crop[0:48,0:48,2:]
                    face_crop = np.expand_dims(face_crop, 0)
                    
                    cf = model.predict(face_crop)
                    
                    score = np.argmax(cf)
                    label = labels[score]
                    idx = 100 * np.max(cf)
                    label = label +" pr:"+ str(int(idx))+"%"
                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(temp, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(temp, format="bgr24")
                #return av.VideoFrame.from_image(temp)

        ctx = webrtc_streamer(
            key="opencv-filter",
            #mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )

        if ctx.video_processor:
            ctx.video_transformer.update_style(style_selection)

            

	
    