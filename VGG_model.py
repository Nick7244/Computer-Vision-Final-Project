import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16 # images are 224×224
from keras.applications.inception_v3 import InceptionV3 # images are 299×299
from keras.applications.resnet50 import ResNet50 # images are 224×224
import numpy as np

# Set up variables
train_data_dir = 'data/train'
test_data_dir = 'data/validation'
num_classes = 43

# VGG16

model = VGG16(include_top=False, input_shape=(300, 300, 3)) # Set shape to whateer is easiest

flat1 = Flatten()(model.outputs)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(num_classes, activation='softmax')(class1)

# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()
# ...



# 



# Use image.ImageDataGenerator to augment data
