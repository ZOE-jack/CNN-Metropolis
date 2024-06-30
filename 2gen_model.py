from keras.datasets import mnist
import numpy as np
import matplotlib.pylab as plt
import json
import tensorflow as tf
import keras 
import time
import os
import visualkeras
import PIL
from PIL import ImageFont
#from tempfile import TemporaryFile
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D, UpSampling2D,Dropout, Conv1D, Dense, Reshape, Flatten
from keras.utils import get_custom_objects
from keras.layers.core import Activation
import neural_structured_learning as nsl
import visualkeras

# Model Structure at first 
model = Sequential()
# encoder network
model.add(Conv2D(name="Conv2d_1", filters = 128, kernel_size = (2,2), activation = 'relu', padding = 'same', input_shape = (64,64,1)))#
model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization1"))
model.add(Conv2D(name="Conv2d_2", filters = 256, kernel_size = (2,2),strides = (3,3), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization2"))
model.add(Conv2D(name="Conv2d_3", filters = 256, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization3"))
model.add(Conv2D(name="Conv2d_4", filters = 512, kernel_size = (2,2),strides = (3,3), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization4"))
model.add(Conv2D(name="Conv2d_5", filters = 512, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization5"))
model.add(Conv2D(name="Conv2d_6", filters = 512, kernel_size = (2,2),strides = (3,3), activation = 'relu', padding = 'same'))
model.add(tf.keras.layers.BatchNormalization(name="BatchNormalization6"))
model.add(Flatten(name="Flatten"))
model.add(Dense(name="Dense1", units=1024, activation = 'relu'))
model.add(Dense(name="Dense2", units=256, activation = 'relu'))
model.add(Dense(name="Dense3", units=64, activation = 'relu'))
model.add(Dense(name="Dense4", units=16))

#keras.models.save_model(filepath="./data/model/model.keras", model=model)

# Plot the visualized figure by visualkeras package
font = ImageFont.load_default()
visualkeras.layered_view(model, legend=True, font=font)


# Model Structure Improvement by regulization
model2 = Sequential()
# encoder network
model2.add(Conv2D(name="Conv2d_1", filters = 128, kernel_size = (2,2), activation = 'relu', padding = 'same', input_shape = (64,64,1)))#
model2.add(tf.keras.layers.BatchNormalization(name="BatchNormalization1"))
model2.add(Conv2D(name="Conv2d_2", filters = 256, kernel_size = (2,2),strides = (3,3), activation = 'relu', padding = 'same'))
model2.add(tf.keras.layers.BatchNormalization(name="BatchNormalization2"))
model2.add(Conv2D(name="Conv2d_3", filters = 256, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model2.add(tf.keras.layers.BatchNormalization(name="BatchNormalization3"))
model2.add(Conv2D(name="Conv2d_4", filters = 512, kernel_size = (2,2),strides = (3,3), activation = 'relu', padding = 'same'))
model2.add(tf.keras.layers.BatchNormalization(name="BatchNormalization4"))
model2.add(Conv2D(name="Conv2d_5", filters = 512, kernel_size = (2,2), activation = 'relu', padding = 'same'))
model2.add(tf.keras.layers.BatchNormalization(name="BatchNormalization5"))
model2.add(Conv2D(name="Conv2d_6", filters = 512, kernel_size = (2,2),strides = (3,3), activation = 'relu', padding = 'same'))
model2.add(tf.keras.layers.BatchNormalization(name="BatchNormalization6"))
model2.add(Flatten(name="Flatten"))
model2.add(Dense(name="Dense1", units=1024,
                 activation = 'relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
model2.add(Dense(name="Dense2", units=256,
                 activation = 'relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
model2.add(Dense(name="Dense3", units=64,
                 activation = 'relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
model2.add(Dense(name="Dense4", units=16))

#keras.models.save_model(filepath="./data/model/model_regular.keras", model=model2)