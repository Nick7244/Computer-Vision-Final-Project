'''
Load a model architecture and image and print probabilities across classes
'''

import keras
from keras.applications import VGG16, InceptionV3, ResNet50
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import pickle
import datetime

# Variables
model_type = "vgg16" # Choose from one of the keys of MODELS
num_class = 5
test_batch_size = 4
test_data_dir = 'Reduced Testing'
num_test = 24 # Corresponds to number of testing images
model_file = 'best_vgg16_25.h5'

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

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    preprocessing_function=preprocess)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size = input_shape,
    batch_size = test_batch_size,
    shuffle = False,
    class_mode='categorical')

model = load_model(model_file)

print('Model loaded')

start_time = datetime.datetime.now()

img_path = 'test_class0.ppm' 

img = load_img(img_path, target_size = input_shape)
img_array = img_to_array(img)
img_pre = preprocess(img_array)

final = np.expand_dims(img_pre, axis=0)

prediction = model.predict(final)

end_time = datetime.datetime.now()

print(prediction)

elapsed_time = end_time - start_time

print(elapsed_time)
start_time = datetime.datetime.now()

img_path = 'test_class13.ppm' 

img = load_img(img_path, target_size = input_shape)
img_array = img_to_array(img)
img_pre = preprocess(img_array)

final = np.expand_dims(img_pre, axis=0)

prediction = model.predict(final)

end_time = datetime.datetime.now()

print(prediction)

elapsed_time = end_time - start_time

print(elapsed_time)
#print(prediction)
