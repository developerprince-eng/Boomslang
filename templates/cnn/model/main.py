from __future__ import absolute_import, division, print_function
from bokeh.plotting import figure, output_file, show
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import RMSProp
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
# import tensorflow as tf currently no support for tensorflow manually install(make sure pc  supports tensorlfow)
import pandas as pd 
import os
from os.path import realpath, abspath
import numpy as np 

os.getcwd()
os.listdir(os.getcwd())

class MODEL():
    def __init__(self):
        self.is_model = True
    
    def depth_2_cnn(self,X_train, Y_train, X_test, Y_test,num_classes,nb_epoch, verbose,validation_split ,batch_size, filterNum, dim1, dim2, img_row, img_col, img_channel):
        model = Sequential()
        model.add(Conv2D(filterNum, (dim1, dim2), padding='same', input_shape=(img_row, img_col, img_channel), activation='relu'))
        model.add(MaxPooling2D(poo_size=(2, 2)))
        model.add(Conv2D((filterNum*2), (dim1, dim2), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512), activation='relu')
        model.add(Dense(num_classes), activation='softmax')
        model.model(loss='categorical_crossentropy', optimizer=RMSProp(), metric=['accuracy'])
        model.summary()

        model.fit(X_train, Y_train , batch_size=batch_size, epochs=nb_epoch, validation_split=validation_split, verbose=verbose)
        score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=verbose)
    
        #Save Model Json
        model_json = model.to_json()
        with open("CNN01_model.json", "w") as json_file:
            json_file.write(model_json)
        #Save Model H5
        model.save_weights("CNN01_model.h5")
        accuracy = score[1] 
        accuracy = accuracy * 100

        return accuracy

    def depth_3_cnn(self,X_train, Y_train, X_test, Y_test,num_classes,nb_epoch, verbose,validation_split ,batch_size, filterNum, dim1, dim2, img_row, img_col, img_channel):
        model = Sequential()
        model.add(Conv2D(filterNum, (dim1, dim2), padding='same', input_shape=(img_row, img_col, img_channel), activation='relu'))
        model.add(MaxPooling2D(poo_size=(4, 4)))
        model.add(Conv2D((filterNum*2), (dim1, dim2), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D((filterNum*4), (dim1, dim2), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512), activation='relu')
        model.add(Dense(num_classes), activation='softmax')
        model.model(loss='categorical_crossentropy', optimizer=RMSProp(), metric=['accuracy'])
        model.summary()

        model.fit(X_train, Y_train , batch_size=batch_size, epochs=nb_epoch, validation_split=validation_split, verbose=verbose)
        score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=verbose)
    
        #Save Model Json
        model_json = model.to_json()
        with open("CNN01_model.json", "w") as json_file:
            json_file.write(model_json)
        #Save Model H5
        model.save_weights("CNN01_model.h5")
        accuracy = score[1] 
        accuracy = accuracy * 100

        return accuracy
