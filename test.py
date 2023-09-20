'''
PyPower Projects
Emotion Detection Using AI
'''

#USAGE : python test.py

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') 
#It willDetect face from input frame
classifier =load_model('./Emotion_Detection.h5') 
#This model will Predict the emotion

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
# 5 Fix classes used for training the model

cap = cv2.VideoCapture(0)
#Captures live image


while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    #Converting it into Grayscale (LBPH Algorithm shown in CASE STUDY)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # The detected face is stored here ..
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    #Creating Blue rectangle around the face.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        #Resizing it according to architecture mobilenet
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        #Converting image into Array
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            ##########################Core Part Starts##########################
            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            # Preds.argmax Function returns the array index of maximum matching image probability
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            ##########################Core Part Ends##########################
            print("\nlabel = ",label)
            label_position = (x,y)
            ##Putting text on Face
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    ##Exit Condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()