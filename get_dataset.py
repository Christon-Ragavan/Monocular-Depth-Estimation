
# import skimage.io as io
import numpy as np
import h5py
import os
from scipy.io import loadmat
# import tables
import matplotlib.pyplot as plt
import utils


def read_dataset_mat_file(file_dir):
    f = h5py.File(file_dir)
    print (f.keys())
    return f


def decode_depth_image(data):
    raw_image = data['images']
    depth_image = data['depths']
    trans_raw_images = np.empty([raw_image.shape[0], 3, 480, 640])
    trans_depth_images = np.empty([depth_image.shape[0], 480, 640])

    # print ("raw_image", raw_image.shape, "\ndepth_image",depth_image.shape )
    # print ("trans_raw_images", trans_raw_images.shape, "\ntrans_depth_images",trans_depth_images.shape )

    for i in range (raw_image.shape[0]):
        trans_raw_images[i, 0, :, :] = raw_image[i, 0, :, :].T
        trans_raw_images[i, 1, :, :] = raw_image[i, 1, :, :].T
        trans_raw_images[i, 2, :, :] = raw_image[i, 2, :, :].T
        trans_depth_images[i, :, :] = depth_image[i, :, :].T

    # op= utils.transformation.raw_image_tansformation(trans_raw_images[0])

    return trans_raw_images, trans_depth_images


def run():
    path_to_depth = '/home/chna1572/workspace/depth_estimation/out/Datasets/nyu_depth_v2_labeled.mat'
    data = read_dataset_mat_file(path_to_depth)
    trans_raw_images, trans_depth_images = decode_depth_image(data)

    return trans_raw_images,trans_depth_images

