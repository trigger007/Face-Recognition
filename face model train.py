# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:38:02 2020

@author: ADITYA SINGH
"""


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Datasets/Train'
valid_path = 'Datasets/Test'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('Dataset/Train/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


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

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from tensorflow.keras.models import load_model

model.save('facefeatures_new_model_vgg16.h5')








def train_mod():
    # dimensions of our images.

    img_width, img_height = 320, 240
    
    train_data_dir = 'data/train'
    
    validation_data_dir = 'data/validation'
    
    nb_train_samples = 200  #total
    
    nb_validation_samples = 10  # total
    
    epochs = 6
    
    batch_size = 10
    
    if K.image_data_format() == 'channels_first':
    
    input_shape = (3, img_width, img_height)
    
    else:
    
    input_shape = (img_width, img_height, 3)
    
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
    
    model.add(Dense(1))
    
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
    
              optimizer='rmsprop',
    
              metrics=['accuracy'])
    
    # this is the augmentation configuration we will use for training
    
    train_datagen = ImageDataGenerator(
    
    rescale=1. / 255,
    
    shear_range=0.2,
    
    zoom_range=0.2,
    
    horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    
    # only rescaling
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
    
    train_data_dir,
    
    target_size=(img_width, img_height),
    
    batch_size=batch_size,
    
    class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
    
    validation_data_dir,
    
    target_size=(img_width, img_height),
    
    batch_size=batch_size,
    
    class_mode='binary')
    
    model.fit_generator(
    
    train_generator,
    
    steps_per_epoch=nb_train_samples // batch_size,
    
    epochs=epochs,
    
    validation_data=validation_generator,
    
    validation_steps=5)
    
    model.save('model.h5')