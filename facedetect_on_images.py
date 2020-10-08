# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:13:37 2020

@author: ADITYA SINGH
"""

import cv2
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('facefeatures_new_model_vgg19.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_extractor(img):

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face



img = cv2.imread('hello1.jpg')




face1=face_extractor(img)

face = cv2.resize(img, (224, 224))
im = Image.fromarray(face, 'RGB')
   #Resizing into 128x128 because we trained the model with this image size.
img_array = np.array(im)
            #Our keras model used a 4D tensor, (images x height x width x channel)
            #So changing dimension 128x128x3 into 1x128x128x3 
img_array = np.expand_dims(img_array, axis=0)
pred = model.predict(img_array)
print(pred)
