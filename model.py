# Import Keras libraries
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from keras.backend import tf as ktf

# Import data augmentation library
from genAugData import *


def resizeImg(img):
    return ktf.image.resize(img, [66,200])

def nvidiaPilotNet():
    model = Sequential()

    # Lambda layer: crop image
    model.add(Lambda(lambda imgs: imgs[:,80:,:,:], input_shape=(160,320,3)))
    # Lambda layer: normalize image by centering the mean at 0
    model.add(Lambda(lambda imgs: (imgs/255.0) - 0.5))

    # A series of 3 convolution layers: 5x5 convolution, 2x2 stride
    # Followed by a ReLU activation layer and a batch normalization layer
    model.add(Convolution2D(24, (5,5), activation="relu", strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(36, (5,5), activation="relu", strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, (5,5), activation="relu", strides=(2,2)))
    model.add(BatchNormalization())

    # A series of 2 convolution layer: 3x3 convolution, 1x1 stride
    # Followed by a ReLU activation layer and a batch normalization layer
    model.add(Convolution2D(64, (3,3), activation="relu", strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (3,3), activation="relu", strides=(2,2)))
    model.add(BatchNormalization())

    # A flattening layer
    model.add(Flatten())

    # A series of 4 fully connected layers
    # Followed by a ReLU activation layer and a batch normalization layer
    model.add(Dense(1164, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(200, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(50, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(10, activation="relu"))
    model.add(BatchNormalization())

    # A output layer
    model.add(Dense(1))

    # Multi-GPU model: to accelerate the model training by utilizing 1 Titan V and 2 Titan XP GPU cards
    multi_model = multi_gpu_model(model, gpus=3)
    multi_model.compile(loss="MSE", optimizer=Adam(lr=0.005))

    return multi_model


