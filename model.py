import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.preprocessing import image
from keras.models import Sequential
import numpy as np

# Set up variables
train_data_dir = 'data/train'
test_data_dir = 'data/validation'


# Set up model architecture


# Use image.ImageDataGenerator to augment data
