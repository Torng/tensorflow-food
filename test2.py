import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from numpy import array
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input, decode_predictions
with open('food2.yaml') as yamlfile:
    loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('weights2.h5')
    
    sgd = Adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    
    path='./testmodel2/'
    timage = []
    for f in os.listdir(path):
        timage = []
        img = image.load_img(path + f, target_size=(128,128))
        img_array = image.img_to_array(img)
        timage.append(img_array)
        imgnp = np.array(timage)
        imgnp /= 255
        result = model.predict(imgnp)
        x = result.argmax(axis=-1)
#        generator = train_datagen.flow_from_directory("train", batch_size=batch_size)
#        x = np.expand_dims(img_array, axis=0)
#        x = preprocess_input(x)
#        result = model.predict_classes(x)
#        classes = np.argmax(result, axis=0)
        print(f, x)
