'''
Before running, make sure all variables in first section are correct
Run with model argument vgg16, resnet, or inception for respective architecture
Uses pretrained weights on ImageNet
Feeds into custom output block of flatten, dense, dropout, dense
Saves model

TODO: Actually train and see which architecture is best

Next steps in other files:
Extract features from previous blocks of architectures and test performance
    as input to current custom output block
Create custom architecture
Evaluate speed of prediction and size of models
'''
import keras
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import argparse
#import cv2
#import random
#import pickle

# Variables
num_class = 43
batch_size = 32
train_data_dir = 'data/train'
validation_data_dir = 'data/valid'
num_train = 1000
num_validation = 200
model_file = 'first_try.h5'

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

# Dictionary of pretrained architectures
MODELS = {"vgg16": VGG16, "inception": InceptionV3, "resnet": ResNet50}

# Makes sure using a valid model
if args["model"] not in MODELS.keys():
	raise AssertionError("Invalid model")

# Gets proper shape and preprocessing function
input_shape = (224,224)
preprocess = imagenet_utils.preprocess_input

if args["model"] == "inception":
	input_shape = (299, 299)
	preprocess = preprocess_input

train_datagen = ImageDataGenerator( 
    rescale=1. / 255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    preprocessing_function=preprocess)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=preprocess)

train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size= input_shape, 
    batch_size=batch_size, 
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size= input_shape,
    batch_size=batch_size,
    class_mode='categorical')


# Initializes model with ImageNet weights and prevent retraining
model_name = MODELS[args["model"]]
model = model_name(include_top=False, weights="imagenet")
for layer in model.layers:
    layer.trainable = False

# New classification
model.add(Flatten()) 
model.add(Dense(1024, activation='relu')) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(num_class)) 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch= num_train // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps= num_validation// batch_size)

model.save(model_file)

'''
To load model, do model = load_model('model.h5')
'''

