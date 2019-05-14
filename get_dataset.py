
# import skimage.io as io
import numpy as np
import h5py
import os
from scipy.io import loadmat
# import tables
import matplotlib.pyplot as plt
import extraction_tools

from sklearn.model_selection import train_test_split




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


    return trans_raw_images, trans_depth_images




def import_nyu_sample_data(do_extract= False, path_to_depth= '/home/chna1572/workspace/depth_estimation/out/Datasets/nyu_depth_v2_labeled.mat'):
    save_dir = '/home/chna1572/workspace/depth_estimation/xtract/Datasets/saved_npy_files/'
    if do_extract:
        print("Extracting...")

        data = read_dataset_mat_file(path_to_depth)
        trans_raw_images, trans_depth_images = decode_depth_image(data)
        print(trans_raw_images.shape, trans_depth_images.shape)

        raw_train, raw_test, depth_train, depth_test = train_test_split(trans_raw_images, trans_depth_images, shuffle=False,
                                                                        test_size=0.3)
        print("dtype:", raw_train.dtype)
        print("RAW", raw_train.shape, raw_test.shape)
        print("DEPTH:", depth_train.shape, depth_test.shape)
        depth_train = depth_train.reshape(depth_train.shape[0], 1, 480, 640).astype('float32')
        depth_test = depth_test.reshape(depth_test.shape[0], 1, 480, 640).astype('float32')

        raw_train = raw_train.reshape(raw_train.shape[0], 3, 480, 640).astype('float32')
        raw_test = raw_test.reshape(raw_test.shape[0], 3, 480, 640).astype('float32')

        raw_train = raw_train / 255.0
        raw_test = raw_test / 255.0
        depth_train = depth_train / 255.0
        depth_test = depth_test / 255.0

        print("RAW", raw_train.shape, raw_test.shape)

        np.save(save_dir+'raw_train.npy', raw_train)
        np.save(save_dir+'raw_test.npy', raw_test)
        np.save(save_dir+'depth_train.npy', depth_train)
        np.save(save_dir+'depth_test.npy', depth_test)
        print("Extraction Saved Successfully!")

    else:
        print("Loading--")
        raw_train = np.load(os.path.join(save_dir, 'raw_train.npy'))
        raw_test = np.load(os.path.join(save_dir, 'raw_test.npy'))
        depth_train = np.load(os.path.join(save_dir, 'depth_train.npy'))
        depth_test = np.load(os.path.join(save_dir, 'depth_test.npy'))
        print("RAW", raw_train.shape, raw_test.shape, depth_train.shape, depth_test.shape)

    # np.savez('features.npz',  raw_train= raw_train,  raw_test =raw_test, depth_train=depth_train, depth_test=depth_test)

    return raw_train, raw_test, depth_train, depth_test
