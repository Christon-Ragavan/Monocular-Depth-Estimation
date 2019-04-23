import numpy as np
import h5py
import os
from scipy.io import loadmat
import tables
import matplotlib.pyplot as plt

class transformation():


    def raw_image_tansformation(raw_image):

        trans_raw_images = np.empty([480, 640,3])
        trans_raw_images[:, :, 0] = raw_image[0, :, :]
        trans_raw_images[:, :, 1] = raw_image[1, :, :]
        trans_raw_images[:, :, 2] = raw_image[2, :, :]
        return trans_raw_images

    def depth_image_tansformation(depth_image):
        trans_depth_images = np.empty([480, 640])
        trans_depth_images = depth_image.T
        return  trans_depth_images

