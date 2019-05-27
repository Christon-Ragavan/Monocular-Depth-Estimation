
# import skimage.io as io
import numpy as np
import h5py
import os
from scipy.io import loadmat
# import tables
import matplotlib.pyplot as plt
from matplotlib.image import imread
import extraction_tools
import glob
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import cv2




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



def crop_images(raw_train, raw_test, depth_train, depth_test):


    raw_train = raw_train[:,:,:100,:100]
    raw_test = raw_test[:,:,:100,:100]
    depth_train = depth_train[:,:,:100,:100]
    depth_test = depth_test[:,:,:100,:100]
    print("CROP TO SIZE OF ---->", raw_train.shape, raw_test.shape, depth_train.shape, depth_test.shape)


    return raw_train, raw_test, depth_train, depth_test



def ploting(trans_raw_images,trans_depth_images ):
    # op = extraction_tools.transformation.raw_image_tansformation(trans_raw_images)
    # print(np.shape(op))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # ax3 = fig.add_subplot(133)

    ax1.imshow(trans_raw_images.reshape(480, 640, 3))
    ax2.imshow(trans_depth_images.reshape(480, 640))
    # ax2.imshow(predictions[0].reshape(480, 640))

    ax1.title.set_text('Original_RGB_image')
    ax2.title.set_text('Ground_truth_Depth_map')
    # ax3.title.set_text('Predicted_Depth_map')

    plt.show()


def import_nyu_sample_data(do_extract= False,
                           crop = True,
                           do_save_features = False,
                           path_to_depth= '/home/chna1572/workspace/depth_estimation/out/Datasets/nyu_depth_v2_labeled.mat'):
    save_dir = '/home/chna1572/workspace/depth_estimation/xtract/Datasets/saved_npy_files/'
    if do_extract:
        print("Extracting...")

        data = read_dataset_mat_file(path_to_depth)
        print("\n\n----1rd step----\n\n")
        trans_raw_images, trans_depth_images = decode_depth_image(data)
        print("MAX val:", np.max(trans_raw_images), np.max(trans_depth_images))
        print("MIN val:", np.min(trans_raw_images), np.min(trans_depth_images))
        print("SHAPE:", np.shape(trans_raw_images), np.shape(trans_depth_images))


        # ploting(trans_raw_images,trans_depth_images )

        trans_raw_images = trans_raw_images / 255.0
        # trans_depth_images = trans_depth_images / 255.0
        print("\n\n----2nd step----\n\n")
        print("MAX val:", np.max(trans_raw_images), np.max(trans_depth_images))
        print("MIN val:", np.min(trans_raw_images), np.min(trans_depth_images))
        print("SHAPE:", np.shape(trans_raw_images), np.shape(trans_depth_images))
        ploting(trans_raw_images, trans_depth_images)

        raw_train, raw_test, depth_train, depth_test = train_test_split(trans_raw_images, trans_depth_images, shuffle=False,
                                                                         test_size=0.3)


        
        depth_train = depth_train.reshape(depth_train.shape[0], 1, 480, 640).astype('float32')
        depth_test = depth_test.reshape(depth_test.shape[0], 1, 480, 640).astype('float32')

        raw_train = raw_train.reshape(raw_train.shape[0], 3, 480, 640).astype('float32')
        raw_test = raw_test.reshape(raw_test.shape[0], 3, 480, 640).astype('float32')
        # raw_train = raw_train / 255.0
        # raw_test = raw_test / 255.0
        # depth_train = depth_train / 255.0
        # depth_test = depth_test / 255.0

        print("\n\n----3rd step----\n\n")
        print("MAX val:", np.max(raw_train), np.max(raw_test),np.max(depth_train), np.max(depth_test))
        print("MIN val:", np.min(raw_train), np.min(raw_test),np.min(depth_train), np.min(depth_test))
        print("SHAPE:", np.shape(raw_train), np.shape(depth_train))




        if do_save_features:

            np.save(save_dir+'raw_train.npy', raw_train)
            np.save(save_dir+'raw_test.npy', raw_test)
            np.save(save_dir+'depth_train.npy', depth_train)
            np.save(save_dir+'depth_test.npy', depth_test)
            print("Extraction Saved Successfully!")

        if crop:

            raw_train, raw_test, depth_train, depth_test = extraction_tools.transformation.crop_images(raw_train,
                                                                                                       raw_test,
                                                                                                       depth_train,
                                                                                                       depth_test,
                                                                                                       DS_factor=2)
        else:
            print("Nothing Here")

    else:
        print("Loading--")
        raw_train = np.load(os.path.join(save_dir, 'raw_train.npy'))
        raw_test = np.load(os.path.join(save_dir, 'raw_test.npy'))
        depth_train = np.load(os.path.join(save_dir, 'depth_train.npy'))
        depth_test = np.load(os.path.join(save_dir, 'depth_test.npy'))
        print("RAW", raw_train.shape, raw_test.shape, depth_train.shape, depth_test.shape)

        if crop:
            print("GOing to crop")
            raw_train, raw_test, depth_train, depth_test = extraction_tools.transformation.crop_images(raw_train,
                                                                                                       raw_test,
                                                                                                       depth_train,
                                                                                                       depth_test,
                                                                                                       DS_factor=2)
            # resize(raw_train, raw_test, depth_train, depth_test)
            print("After Cropping", raw_train.shape, raw_test.shape, depth_train.shape, depth_test.shape)
        else:
            print("Nothing Here")

    # np.savez('features.npz',  raw_train= raw_train,  raw_test =raw_test, depth_train=depth_train, depth_test=depth_test)

    return raw_train, raw_test, depth_train, depth_test

def import_nyu_dir(do_extract= False,
                           crop = True,
                           do_save_features = False,
                           path_to_dir= '/home/chna1572/workspace/depth_estimation/xtract/Datasets/nyu_datasets'):
    save_dir = '/home/chna1572/workspace/depth_estimation/xtract/Datasets/saved_npy_files/'





    if do_extract:
        print("Extracting...")

        raw_images_all, depth_images_all = [],[]

        raw_list, depth_list = [], []
        print(os.path.isdir(path_to_dir))

        raw_list.extend(glob.glob(os.path.join(path_to_dir, '*.jpg')))
        depth_list.extend(glob.glob(os.path.join(path_to_dir, '*.png')))

        raw_list.sort()
        depth_list.sort()

        print("all_fn_img", len(raw_list),len(depth_list))

        assert len(raw_list) == len(depth_list)

        for index in range (len(depth_list)):
            # curr_raw = load_img(raw_list[index], target_size=(128, 128))
            # curr_raw = img_to_array(curr_raw)

            curr_raw = imread(raw_list[index])
            curr_raw = curr_raw /255.0
            raw_images_all.append(curr_raw)

            curr_depth = imread(depth_list[index])
            depth_images_all.append(curr_depth)

            # ploting(curr_raw, curr_depth)
            print(index, np.shape(curr_raw), np.shape(curr_depth), np.max(curr_raw), np.max(curr_depth), np.min(curr_raw), np.min(curr_depth))

        # print(np.shape(raw_images_all), np.shape(depth_images_all),  np.max(raw_images_all), np.max(depth_images_all), np.min(raw_images_all), np.min(depth_images_all))

        raw_images_all = np.asarray(raw_images_all, dtype=np.float32)
        depth_images_all = np.asarray(depth_images_all, dtype=np.float32)
        # print(raw_images_all.shape[0])
        raw_images_all = raw_images_all.reshape(raw_images_all.shape[0],  480, 640,3).astype('float32')
        depth_images_all = depth_images_all.reshape(depth_images_all.shape[0], 480, 640,1).astype('float32')


        print(np.shape(raw_images_all), np.shape(depth_images_all),  np.max(raw_images_all), np.max(depth_images_all), np.min(raw_images_all), np.min(depth_images_all))
        raw_train, raw_test, depth_train, depth_test = train_test_split(raw_images_all, depth_images_all,
                                                                        shuffle=False,
                                                                        test_size=0.3)

        print("\n\n----3rd step----\n\n")
        print("MAX val:", np.max(raw_train), np.max(raw_test), np.max(depth_train), np.max(depth_test))
        print("MIN val:", np.min(raw_train), np.min(raw_test), np.min(depth_train), np.min(depth_test))
        print("SHAPE:", np.shape(raw_train), np.shape(depth_train))

        if do_save_features:

            np.save(save_dir+'raw_train_f.npy', raw_train)
            np.save(save_dir+'raw_test_f.npy', raw_test)
            np.save(save_dir+'depth_train_f.npy', depth_train)
            np.save(save_dir+'depth_test_f.npy', depth_test)
            print("Extraction Saved Successfully!")


    else:
        print("Loading--")
        raw_train = np.load(os.path.join(save_dir, 'raw_train_f.npy'))
        raw_test = np.load(os.path.join(save_dir, 'raw_test_f.npy'))
        depth_train = np.load(os.path.join(save_dir, 'depth_train_f.npy'))
        depth_test = np.load(os.path.join(save_dir, 'depth_test_f.npy'))
        print("RAW", raw_train.shape, raw_test.shape, depth_train.shape, depth_test.shape)

        # ploting(raw_train[0], depth_train[0])
        # ploting(depth_train[0], depth_test[0])

    if crop:
        print("GOing to crop")
        raw_train, raw_test, depth_train, depth_test = extraction_tools.transformation.crop_images(raw_train,
                                                                                                   raw_test,
                                                                                                   depth_train,
                                                                                                   depth_test,
                                                                                                   DS_factor=2)

        # ploting(raw_train[0], depth_train[0])
        # ploting(depth_train[0], depth_test[0])

    return raw_train, raw_test, depth_train, depth_test
