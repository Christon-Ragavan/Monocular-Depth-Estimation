
# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2DTranspose, Conv3DTranspose
from keras.utils import np_utils
from keras.layers.core import Lambda
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import get_dataset
import utils


K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


trans_raw_images,trans_depth_images = get_dataset.run()
print (trans_raw_images.shape, trans_depth_images.shape)


raw_train, raw_test, depth_train, depth_test = train_test_split(trans_raw_images, trans_depth_images, shuffle=False,test_size=0.3)
print ("dtype:",raw_train.dtype)
print ("RAW", raw_train.shape, raw_test.shape)
print ("DEPTH:", depth_train.shape, depth_test.shape)

depth_train = depth_train.reshape(depth_train.shape[0], 1, 480, 640).astype('float32')
depth_train = depth_train.reshape(depth_train.shape[0], 1, 480, 640).astype('float32')


# raw_train_1 = utils.transformation.raw_image_tansformation(raw_train[0])
# raw_test_1 = utils.transformation.raw_image_tansformation(raw_test[0])
# plt.subplot(221)
# plt.imshow(raw_train_1/255.0)
# plt.subplot(222)
# plt.imshow(raw_test_1/255.0)
# plt.subplot(223)
# plt.imshow(depth_train[0])
# plt.subplot(224)
# plt.imshow(depth_test[0])
# # show the plot
# plt.show()


raw_train = raw_train/255.0
raw_test = raw_test/255.0
print ("RAW", raw_train.shape, raw_test.shape)
# depth_train = raw_train
# depth_test = raw_test


def de_conv():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1,480, 640), activation='relu'))
    model.add(Conv2D(15, (3, 3), activation='relu'))

    model.add(Conv2DTranspose(15, (3,3), activation='relu'))
    model.add(Conv2DTranspose(1, (5,5), activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model



print("building model...")
model = de_conv()

# # Fit the model
model.fit(depth_train, depth_train, validation_data=(depth_test, depth_test), epochs=1)