# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
import coremltools
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.applications.resnet50 import preprocess_input, decode_predictions
nb_classes = 2
def load_data():
    path = './food3/'
    files = os.listdir(path)
    images = []
    labels = []
    clabel = []
    files.sort()
    for f1 in files:
        imgfile = path + f1
        files2 = os.listdir(imgfile)
        newf1 = f1[2:]
        clabel.append(newf1)
        print(f1)
        for f in files2:
            img_path = imgfile + '/' + f
            img = image.load_img(img_path, target_size=(400,400,3))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(f1)
    data = np.array(images)
    labels = np.array(labels)
#    labels = np_utils.to_categorical(labels, 2)
    encoder = LabelBinarizer()
    newlabel = encoder.fit_transform(labels)
#    onehot_encoder = OneHotEncoder(sparse=False)
#    newlabel = newlabel.reshape((len(newlabel)),1)
#    onehot_encoded = onehot_encoder.fit_transform(newlabel)
    return data, newlabel ,clabel

print("train.......")

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(400,400,3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, activation='softmax'))
model.summary()
images, labels , clabel= load_data()
#labels = np_utils.to_categorical(labels, 2)
images /= 255
x_train, x_test, y_train,y_test = train_test_split(images, labels, test_size=0.2)
sgd = Adam(lr=0.0003)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
gen = ImageDataGenerator(featurewise_center=False,
                         samplewise_center=False,
                         horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=8,
                         shear_range=0.3,
                         zoom_range=0.08,
                         data_format="channels_last")
gen.fit(x_train)
train_generator = gen.flow(x_train, y_train, batch_size=50)
test_gen = ImageDataGenerator(data_format="channels_last")
test_gen.fit(x_test)
test_generator = test_gen.flow(x_test, y_test, batch_size=300)
model.fit_generator(train_generator,epochs=10, verbose=1, validation_data=(x_test, y_test))
scroe, accuracy = model.evaluate(x_test, y_test, batch_size=200)
print('scroe:', scroe, 'accuracy:', accuracy)
#tlabel = ['3cup','bigw']
yaml_string = model.to_yaml()
with open('./food2.yaml', 'w') as outfile:
    outfile.write(yaml_string)
model.save_weights('./weights2.h5')
#coreml_model = coremltools.converters.keras.convert(model,
#                                                    input_names="image",
#                                                    image_input_names="image",
#                                                    output_names="foodname",
#                                                    class_labels=clabel,
#                                                    image_scale=1/255.0,
#                                                    is_bgr=False
#                                                    )
#
#coreml_model.save('testmodel.mlmodel')

#path='./testmodel/'
#timage = []
#for f in os.listdir(path):
#    timage = []
#    img = image.load_img(path + f, target_size=(128,128))
#    img_array = image.img_to_array(img)
#    timage.append(img_array)
#    imgnp = np.array(timage)
#    imgnp /= 255
#    result = model.predict_classes(imgnp)
#    print(f,result)



