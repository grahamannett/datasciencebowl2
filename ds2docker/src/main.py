import os
import csv
import sys
import numpy as np
import pydicom
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def get_lenet():
    source = (source - 128) * (1.0 / 128)

    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(
        Convolution2D(32, 3, 3, border_mode='full', input_shape=(3, 100, 100)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


def encode_label(label_data):
    """Run encoding to encode the label into the CDF target.
    """
    stytole = label_data[:, 1]
    diastole = label_data[:, 2]
    stytole_encode = np.array([
        (x < np.arange(600)) for x in stytole
    ], dtype=np.uint8)
    diastole_encode = np.array([
        (x < np.arange(600)) for x in diastole
    ], dtype=np.uint8)
    return stytole_encode, diastole_encode


def encode_csv(label_csv, stytole_csv, diastole_csv):
    stytole_encode, diastole_encode = encode_label(
        np.loadtxt(label_csv, delimiter=","))
    np.savetxt(stytole_csv, stytole_encode, delimiter=",", fmt="%g")
    np.savetxt(diastole_csv, diastole_encode, delimiter=",", fmt="%g")

encode_csv("./train-label.csv", "./train-stytole.csv", "./train-diastole.csv")
