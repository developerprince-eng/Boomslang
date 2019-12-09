
from __future__ import absolute_import, division, print_function
from bokeh.plotting import figure, output_file, show
from keras.models import Sequential
from keras.layers import Dense
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

    #Train RNN Classifier
    def kr_train_RNN_Model(self):
        metrics = []
        return metrics
    
    #Train CNN Classifier
    def kr_train_CNN_Model(self):
        metrics = []
        return metrics

    def kr_train_DNN_Seq_01(self,x_dim ,features_train ,features_test, labels_train , labels_test, batch_size):
        # create model
        model = Sequential()
        model.add(Dense(10, input_dim=x_dim, init='uniform', activation='relu'))
        model.add(Dense(60, init='uniform', activation='relu'))
        model.add(Dense(30, init='uniform', activation='sigmoid'))
        model.add(Dense(1, init='uniform', activation='sigmoid'))                                                                                                                                                                                                                       
        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer='adam',
        metrics=['accuracy'])
       
        # Model Summary
        model.summary()
        # Fit the model
        model.fit(features_train, labels_train, epochs=100, batch_size=batch_size, verbose=2)
        model_yaml = model.to_yaml()
        with open("seq01_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model_json = model.to_json()
        with open("seq01_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("seq01_model.h5")
        score = model.evaluate(features_test, labels_test, verbose=1)
        # round predictions
        accuracy = score[1]
        accuracy = accuracy * 100
    
        return accuracy

    
    def kr_train_DNN_Seq_02(self,x_dim ,features_train ,features_test, labels_train , labels_test, batch_size):
        # create model
        model = Sequential()
        model.add(Dense(9, input_dim=x_dim, init='uniform', activation='relu'))
        model.add(Dense(36, init='uniform', activation='relu'))
        model.add(Dense(36, init='uniform', activation='sigmoid'))
        model.add(Dense(1, init='uniform', activation='sigmoid'))                                                                                                                                                                                                                       
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam',
        metrics=['accuracy'])

        # Model Summary
        model.summary()
        # Fit the model
        model.fit(features_train, labels_train, epochs=100, batch_size=batch_size, verbose=2)
        model_yaml = model.to_yaml()
        with open("seq02_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model_json = model.to_json()
        with open("seq02_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("seq02_model.h5")
        score = model.evaluate(features_test, labels_test, verbose=1)
        # round predictions
        accuracy = score[1]
        accuracy = accuracy * 100
    
        return accuracy

    def kr_train_DNN_Seq_03(self,x_dim ,features_train ,features_test, labels_train , labels_test, batch_size):
        # create model
        model = Sequential()
        model.add(Dense(28, input_dim=x_dim, init='uniform',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.005), activation='relu'))
        model.add(Dense(14, init='uniform',kernel_regularizer=regularizers.l2(0.015),activity_regularizer=regularizers.l1(0.02), activation='relu'))
        model.add(Dense(8, init='uniform', activation='relu'))
        model.add(Dense(10, init='uniform', activation='relu'))
        model.add(Dense(10, init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform', activation='relu'))                                                                                                                                                                                                                
        # Compile model
        model.compile(loss='mean_squared_logarithmic_error', optimizer='adam',
        metrics=['accuracy'])
        # checkpoint
        filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        # Model Summary
        model.summary()
        # Fit the model
        history = model.fit(features_train, labels_train,validation_split=0.2 ,epochs=100, batch_size=batch_size,callbacks=callbacks_list, verbose=2)
        model_yaml = model.to_yaml()
        with open("seq03_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model_json = model.to_json()
        with open("seq03_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("seq03_model.h5")
        score = model.evaluate(features_test, labels_test, verbose=1)
        # round predictions
        accuracy = score[1]
        accuracy = accuracy * 100
        #output_file("line_plot.html")
        #line_graph = figure()
        #line_graph.line([history.history['acc']], [history.history['val_acc']])
        #show(line_graph)

        #Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        #Plot training & validation loss values
        plt2.plot(history.history['loss'])
        plt2.plot(history.history['val_loss'])
        plt2.title('Model loss')
        plt2.ylabel('Loss')
        plt2.xlabel('Epoch')
        plt2.legend(['Train', 'Test'], loc='upper left')
        plt2.show()
        return accuracy
 
