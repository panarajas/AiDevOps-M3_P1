# Code to create model by reading the pickle files

# Expects runtime argument - directory where training pickle files are located
# e.g. python Train.py /dir/to/pickle_files

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Conv2D
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.preprocessing.image import ImageDataGenerator
import time
from IPython.display import display 
#from PIL import Image
from keras.utils import multi_gpu_model
import tensorflow as tf
import keras
from PIL import Image, ImageFilter
from tensorflow.keras import backend as k

config = tf.ConfigProto()
 
tf.keras.backend.set_session(tf.Session(config=config))
starttime = time.time()

# dimensions of our images.
img_width, img_height = 60,320#720, 1280#180, 320

train_data_dir = 'pickle/'
#validation_data_dir = 'C:/test'
validation_data_dir = 'pickle/'

import sys
# directory provided at runtime
print('Directory given in commandline', sys.argv[-1])
train_data_dir = sys.argv[-1]
validation_data_dir = sys.argv[-1]
print('train_data_dir', train_data_dir)

nb_train_samples = 10#70000/100
nb_validation_samples = 100
nb_epoch = 1
batch_size = 100

# output_dir = "/opt/ml/model"

input_img = Input(shape=( img_width, img_height,3))

#with tf.device('/gpu:0'):
x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
print(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
print(x)
x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

#with tf.device('/cpu:0'):
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.models import model_from_json
import numpy as np
import os
import cv2
import skimage 
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import pickle as pk


def generate_arrays_from_file(flag):
    if flag == True:
        path = train_data_dir
        print('path updated to (if case) ', path)
    else:
        path = validation_data_dir
        print('path updated to (else case) ', path)
        
    print('Using directory', path)
    imagesName = os.listdir(path)
    print(imagesName)
    print("*************************generate_arrays_from_file")
    while(1):
        count = 1
        for image in imagesName:
            filename = path+"/"+image
            
            file = open(filename,"rb")
            data = pk.load(file)
            print(data.shape)
            data = data.reshape(data.shape[0],60,320,3)
            print(data.shape)
            count =count + 1
            print("Count:",count)
            yield(data,data)
            
autoencoder.fit_generator(
    generate_arrays_from_file(0),
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=generate_arrays_from_file(0),
    nb_val_samples=nb_validation_samples
    )    

print("time...", time.time()-starttime)

model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")
