import numpy as np
import h5py
import os
from scipy.io import loadmat
# import tables
import matplotlib.pyplot as plt

class transformation():


    def raw_image_tansformation(raw_image):

        """
        input saphae
        :return:
        """
        print("Inputshape:",raw_image.shape)
        width = raw_image.shape[1]
        height = raw_image.shape[2]
        channel = raw_image.shape[0]
        trans_raw_images = np.empty([width, height,channel])
        trans_raw_images[:, :, 0] = raw_image[0, :, :]
        trans_raw_images[:, :, 1] = raw_image[1, :, :]
        trans_raw_images[:, :, 2] = raw_image[2, :, :]

        print("trans_raw_images:", trans_raw_images.shape)
        return trans_raw_images

    def depth_image_tansformation(depth_image):
        trans_depth_images = depth_image.T
        return  trans_depth_images
    #
    def crop_images(raw_train, raw_test, depth_train, depth_test, DS_factor = 2):
        DS_factor = 128
        raw_train = raw_train[:, :DS_factor, :DS_factor,:]
        raw_test = raw_test[:, :DS_factor, :DS_factor, : ]
        depth_train = depth_train[:, :DS_factor, :DS_factor]
        depth_test = depth_test[:,  :DS_factor, :DS_factor]
        print("CROP TO SIZE OF ---->", raw_train.shape, raw_test.shape, depth_train.shape, depth_test.shape)

        return raw_train, raw_test, depth_train, depth_test


    # def crop_images(raw_train, raw_test, depth_train, depth_test, DS_factor = 2):
    #     raw_train = raw_train[:, :, ::DS_factor, ::DS_factor]
    #     raw_test = raw_test[:, :, ::DS_factor, ::DS_factor]
    #     depth_train = depth_train[:, :, ::DS_factor, ::DS_factor]
    #     depth_test = depth_test[:, :, ::DS_factor, ::DS_factor]
    #     print("CROP TO SIZE OF ---->", raw_train.shape, raw_test.shape, depth_train.shape, depth_test.shape)
    #
    #     return raw_train, raw_test, depth_train, depth_test
    # def crop_images(raw_train, raw_test, depth_train, depth_test, DS_factor = 2):
    #     import cv2
    #     thumbnail = cv.resize(im, (width / 10, height / 10), interpolation=cv.INTER_AREA)
    #
    #     return raw_train, raw_test, depth_train, depth_test
    #

class utils():

    def get_model_memory_usage(batch_size, model):
        import numpy as np
        from keras import backend as K

        shapes_mem_count = 0
        for l in model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

        number_size = 4.0
        if K.floatx() == 'float16':
            number_size = 2.0
        if K.floatx() == 'float64':
            number_size = 8.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes

