


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
from keras.applications.vgg16 import VGG16
import keras

from collections import defaultdict, OrderedDict
from keras.models import Model
"""

"""


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    return c


def UNet_downloded(image_size):
    import keras
    f = [16, 32, 64, 128, 256]
    inputs = Input(image_size)

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    return model

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def u_net(input_shape):
    # create model
    model = Sequential()
    model.add(Conv2D(3, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(6, (3, 3), activation='relu'))
    model.add(Conv2DTranspose(6, (3,3), activation='relu'))
    model.add(Conv2DTranspose(1, (5,5), activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def u_net_2(input_shape):
    # create model
    model = Sequential()
    model.add(Conv2D(3, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(6, (3, 3), activation='relu'))
    model.add(Conv2DTranspose(6, (3, 3), activation='relu'))
    model.add(Conv2DTranspose(1, (5, 5), activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def create_model_v1(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    #
    #
    # up1 = Conv2D(5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(pool4))
    # upconv1 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # up2 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv1))
    upconv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    upconv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv2)


    up3 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv2))
    upconv3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up3)
    upconv3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv3)
    up4 = Conv2D(4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv3))
    upconv4 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    upconv4 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv4)

    conv_out = Conv2D(1, 1, activation='sigmoid')(upconv4)

    model = Model(input=inputs, output=conv_out)

    # model.compile(optimizer="rmsprop", loss=root_mean_squared_error, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def create_model_vgg_b1(input_shape):
    from segmentation_models import Unet
    from segmentation_models.backbones import get_preprocessing
    from segmentation_models.losses import bce_jaccard_loss
    from segmentation_models.metrics import iou_score

    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)

    model = Unet(BACKBONE, encoder_weights='imagenet')
    model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
    return model

def create_model_v22(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    #
    #
    # up1 = Conv2D(5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(pool4))
    # upconv1 = Conv2D(5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # up2 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv1))
    upconv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    upconv2 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv2)


    up3 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv2))
    upconv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up3)
    upconv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv3)
    up4 = Conv2D(63, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv3))
    upconv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up4)
    upconv4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv4)

    conv_out = Conv2D(1, 1, activation='sigmoid')(upconv4)

    model = Model(input=inputs, output=conv_out)

    model.compile(optimizer="rmsprop", loss=root_mean_squared_error, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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