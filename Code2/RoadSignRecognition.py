#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* OpenCV keras traffic sign recognition
* https://data-flair.training/blogs/python-project-traffic-signs-recognition/
***********************************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os, sys
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML():
    main()
    
#************************************
# Main
#************************************
def main():
    
    num_classes = 43
    epochs = 3
    
    X_train, X_test, y_train, y_test = readLearnData(num_classes, "Train")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    #Converting the labels into one hot encoding
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)

    model = cnn_model(num_classes, X_train.shape[1:])
    
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    model.save("my_model.h5")
    
    #if dP.makeQuantizedTFlite:
    makeQuantizedTFmodel(X_train, model, "name")

    plotResults(history)

    #testing accuracy on test dataset
    from sklearn.metrics import accuracy_score

    y_test = pd.read_csv('Test.csv')

    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    data=[]

    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))

    X_test=np.array(data)

    #pred = model.predict_classes(X_test)
    pred = np.argmax(model.predict(X_test), axis=-1)

    #Accuracy with the test data
    from sklearn.metrics import accuracy_score
    print(accuracy_score(labels, pred))

    model.save("traffic_classifier.h5")


#************************************
# Supporting methods
#************************************
def cnn_model(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    #Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def readLearnData(num_classes, type):
    cur_path = os.getcwd()
    data = []
    labels = []

    #Retrieving the images and their labels
    for i in range(num_classes):
        path = os.path.join(cur_path,type,str(i))
        images = os.listdir(path)

        for a in images:
            try:
                image = Image.open(path + '/'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                #sim = Image.fromarray(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    #Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape, labels.shape)
    #Splitting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
    
    
def plotResults(history):
    #plotting graphs for accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
#************************************
### Create Quantized tflite model
#************************************
def makeQuantizedTFmodel(A, model, name):
    import tensorflow as tf
    print("\n  Creating quantized TensorFlowLite Model...\n")
    
    A2 = tf.cast(A, tf.float32)
    A = tf.data.Dataset.from_tensor_slices((A2)).batch(1)
    
    def representative_dataset_gen():
        for input_value in A.take(100):
            yield[input_value]
            
    #converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(dP.model_name)    # TF2.0-2.2 (will be deprecated)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)    # TF2.3 and higher only for full EdgeTPU support.

    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()

    with open("name"+'.tflite', 'wb') as o:
        o.write(tflite_quant_model)
    
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

