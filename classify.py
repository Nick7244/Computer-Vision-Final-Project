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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import numpy as np
import argparse
import pickle

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

# Variables
num_class = 43
train_batch_size = 48
valid_batch_size = 48
train_data_dir = 'GTSRB/Training'
validation_data_dir = 'GTSRB/Validation'
testing_data_dir = ''
num_train = 21312
num_validation = 5328
num_epochs = 25
model_file = 'first_try_'+args["model"]+'.h5'

# Gets proper shape and preprocessing function
input_shape = (224,224)
preprocess = imagenet_utils.preprocess_input

if args["model"] == "inception":
	input_shape = (299, 299)
	preprocess = preprocess_input

train_datagen = ImageDataGenerator( 
    rescale=1. / 255,
    width_shift_range=[-0.2, 0.2],
    height_shift_range=[-0.2, 0.2],
    brightness_range=[0.5, 1.5],
    zoom_range=[0.75, 1.25],
    shear_range=0.2, 
    horizontal_flip=True,
    preprocessing_function=preprocess)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=[-0.2, 0.2],
    height_shift_range=[-0.2, 0.2],
    brightness_range=[0.5, 1.5],
    zoom_range=[0.75, 1.25],
    shear_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess)

train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size= input_shape, 
    batch_size=train_batch_size, 
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size= input_shape,
    batch_size=valid_batch_size,
    class_mode='categorical')

# Initializes model with ImageNet weights and prevent retraining
model_name = MODELS[args["model"]]
base_model = model_name(include_top=False, weights="imagenet")
for layer in base_model.layers:
    layer.trainable = False

print('Model loaded.')

# New classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Flatten(input_shape = base_model.output_shape[1:])(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_class, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

'''
model.add(Flatten()) 
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(num_class, activation='softmax')) 
'''

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch= num_train // train_batch_size,
        epochs= num_epochs,
        validation_data=validation_generator,
        validation_steps= num_validation // valid_batch_size)

model.save(model_file)

model.evaluate_generator(generator=valid_generator,
        steps=  num_validation // valid_batch_size)

'''
To load model, do model = load_model('model.h5')
'''

