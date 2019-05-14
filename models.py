


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.core import Lambda
from keras.models import Sequential, Model, Input
from keras.layers import Conv2DTranspose, Conv3DTranspose, concatenate, MaxPooling1D, MaxPooling2D, MaxPool2D, UpSampling2D, BatchNormalization, Activation
from keras.models import *
from keras.layers import *
from keras.optimizers import *


"""

"""





def u_net():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(3,480, 640), activation='relu'))
    model.add(Conv2D(15, (3, 3), activation='relu'))

    model.add(Conv2DTranspose(15, (3,3), activation='relu'))
    model.add(Conv2DTranspose(1, (5,5), activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def create_model_v1():
    inputs = Input((3, 480, 640))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    # conv3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # # conv3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    #
    #
    # up1 = Conv2D(5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(pool4))
    # upconv1 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    # up2 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv1))
    # upconv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    # upconv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv2)


    # up3 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(pool2))
    upconv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # upconv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv3)
    up4 = Conv2D(63, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv3))
    upconv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    # upconv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv4)

    conv_out = Conv2D(1, 1, activation='sigmoid')(upconv4)

    model = Model(input=inputs, output=conv_out)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def create_model_v2():
    inputs = Input((3,480, 640))
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(1, 1))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model






def create_model_v4():

        input_size = (3,480, 640)

        inputs = Input(shape=input_size)
        # 128

        down1 = Conv2D(32, (3, 3), padding='same')(inputs)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(32, (3, 3), padding='same')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
        # 64

        down2 = Conv2D(64, (3, 3), padding='same')(down1_pool)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2 = Conv2D(64, (3, 3), padding='same')(down2)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
        # 32

        down3 = Conv2D(128, (3, 3), padding='same')(down2_pool)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3 = Conv2D(128, (3, 3), padding='same')(down3)
        down3 = BatchNormalization()(down3)
        down3 = Activation('relu')(down3)
        down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
        # 16

        down4 = Conv2D(256, (3, 3), padding='same')(down3_pool)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4 = Conv2D(256, (3, 3), padding='same')(down4)
        down4 = BatchNormalization()(down4)
        down4 = Activation('relu')(down4)
        down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
        # 8

        center = Conv2D(512, (3, 3), padding='same')(down4_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(512, (3, 3), padding='same')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        # center

        up4 = UpSampling2D((2, 2))(center)
        up4 = concatenate([down4, up4], axis=3)
        up4 = Conv2D(256, (3, 3), padding='same')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        up4 = Conv2D(256, (3, 3), padding='same')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        up4 = Conv2D(256, (3, 3), padding='same')(up4)
        up4 = BatchNormalization()(up4)
        up4 = Activation('relu')(up4)
        # 16

        up3 = UpSampling2D((2, 2))(up4)
        up3 = concatenate([down3, up3], axis=3)
        up3 = Conv2D(128, (3, 3), padding='same')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        up3 = Conv2D(128, (3, 3), padding='same')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        up3 = Conv2D(128, (3, 3), padding='same')(up3)
        up3 = BatchNormalization()(up3)
        up3 = Activation('relu')(up3)
        # 32

        up2 = UpSampling2D((2, 2))(up3)
        up2 = concatenate([down2, up2], axis=3)
        up2 = Conv2D(64, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(64, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(64, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        # 64

        up1 = UpSampling2D((2, 2))(up2)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(32, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(32, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(32, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        # 128

        classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

        model = Model(inputs=inputs, outputs=classify)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy')

        model.summary()
        return model