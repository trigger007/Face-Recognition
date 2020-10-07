# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:58:26 2020

@author: ADITYA SINGH
"""


import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_extraction(img):
    faces=face_classifier.detectMultiScale(img,1.3, 5)
    
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face=img[y:y+h+50,x:x+w+50]
    
    return cropped_face
   
cap=cv2.VideoCapture(0)
count=0

while True:
    ret, frame=cap.read()
    if face_extraction(frame) is not None:
        count=count+1
        face=cv2.resize(face_extraction(frame),(400,400))
        #face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_path='./Images/'+str(count)+'.jpg'
        cv2.imwrite(file_path, face)
        
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass
    if cv2.waitKey(1)==13 or count==100:#13 is enter key
        break
cap.release()
cv2.destroyAllWindows()
print("collecting Samples Complete")
        
        