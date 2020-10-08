# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 23:10:08 2020

@author: ADITYA SINGH
"""


from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

IMAGE_SIZE = [224, 224]

train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

folders = glob('Dataset/Train/*')

input_shape=IMAGE_SIZE + [3]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(len(folders)))

model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',

          optimizer='rmsprop',

          metrics=['accuracy'])

model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Dataset/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Dataset/Test',
                                           target_size = (224, 224), 
                                           batch_size = 32,
                                           class_mode = 'categorical')



r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=15,
  steps_per_epoch=len(training_set))

import tensorflow as tf

from tensorflow.keras.models import load_model

model.save('facerecog.h5')
# def train_mod():
#     # dimensions of our images.


    
#     nb_train_samples = 200  #total
    
#     nb_validation_samples = 10  # total
    
#     epochs = 6
    
#     batch_size = 10
    
#     if K.image_data_format() == 'channels_first':
    
#         input_shape = (3, img_width, img_height)
    
#     else:
    
#         input_shape = (img_width, img_height, 3)
    
#     model = Sequential()
    
#     model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    
#     model.add(Activation('relu'))
    
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(Conv2D(32, (3, 3)))
    
#     model.add(Activation('relu'))
    
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(Conv2D(64, (3, 3)))
    
#     model.add(Activation('relu'))
    
#     model.add(MaxPooling2D(pool_size=(2, 2)))
    
#     model.add(Flatten())
    
#     model.add(Dense(64))
    
#     model.add(Activation('relu'))
    
#     model.add(Dropout(0.5))
    
#     model.add(Dense(1))
    
#     model.add(Activation('sigmoid'))
    
#     model.compile(loss='binary_crossentropy',
    
#               optimizer='rmsprop',
    
#               metrics=['accuracy'])
    
#     # this is the augmentation configuration we will use for training
    
#     train_datagen = ImageDataGenerator(
    
#     rescale=1. / 255,
    
#     shear_range=0.2,
    
#     zoom_range=0.2,
    
#     horizontal_flip=True)
    
#     # this is the augmentation configuration we will use for testing:
    
#     # only rescaling
    
#     test_datagen = ImageDataGenerator(rescale=1. / 255)
    
#     train_generator = train_datagen.flow_from_directory(
    
#     train_data_dir,
    
#     target_size=(img_width, img_height),
    
#     batch_size=batch_size,
    
#     class_mode='binary')
    
#     validation_generator = test_datagen.flow_from_directory(
    
#     validation_data_dir,
    
#     target_size=(img_width, img_height),
    
#     batch_size=batch_size,
    
#     class_mode='binary')
    
#     model.fit_generator(
    
#     train_generator,
    
#     steps_per_epoch=nb_train_samples // batch_size,
    
#     epochs=epochs,
    
#     validation_data=validation_generator,
    
#     validation_steps=5)
    
#     model.save('model.h5')