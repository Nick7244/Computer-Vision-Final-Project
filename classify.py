'''
Before running, make sure all variables in first section are correct
Run with vgg16, resnet, or inception for respective architecture
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
from keras.applications import VGG16, InceptionV3, ResNet50
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle

# Variables
model_type = "vgg16" # Choose from one of the keys of MODELS
num_class = 43
train_batch_size = 48
valid_batch_size = 48
train_data_dir = 'GTSRB/Training'
validation_data_dir = 'GTSRB/Validation'
testing_data_dir = ''
num_train = 21312 # Corresponds to number of training images
num_validation = 5328 # Corresponds to number of validation images
num_epochs = 25
model_file = 'first_try_'+model_type+'.h5'

# Dictionary of pretrained architectures
MODELS = {"vgg16": VGG16, "inception": InceptionV3, "resnet": ResNet50}

# Makes sure using a valid model
if model_type not in MODELS.keys():
        raise AssertionError("Invalid model")

# Gets proper shape and preprocessing function
input_shape = (224,224)
preprocess = imagenet_utils.preprocess_input

if model_type == "inception":
	input_shape = (299, 299)
	preprocess = preprocess_input

train_datagen = ImageDataGenerator( 
    rescale=1. / 255,
    width_shift_range=[-0.2, 0.2],
    height_shift_range=[-0.2, 0.2],
    brightness_range=[0.5, 1.5],
    zoom_range=[0.75, 1.25],
    shear_range=0.2, 
    preprocessing_function=preprocess)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=[-0.2, 0.2],
    height_shift_range=[-0.2, 0.2],
    brightness_range=[0.5, 1.5],
    zoom_range=[0.75, 1.25],
    shear_range=0.2,
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
model_name = MODELS[model_type]
base_model = model_name(include_top=False, weights="imagenet")
for layer in base_model.layers:
    layer.trainable = False

print('Base model loaded.')

# New classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_class, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath= model+'.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)

model.fit_generator(
        train_generator,
        steps_per_epoch= num_train // train_batch_size,
        epochs= num_epochs,
        validation_data=validation_generator,
        validation_steps= num_validation // valid_batch_size,
	callbacks=[checkpointer])

model.save(model_file)

# Not too sure about this but we'll see when we get here
model.evaluate_generator(generator=valid_generator,
        steps=  num_validation // valid_batch_size)

'''
To load model, do model = load_model('model.h5')
'''
