
import skimage.io as io
import numpy as np
import h5py
import os
from scipy.io import loadmat
import tables
import matplotlib.pyplot as plt

#
# print ("Start here")
#
# # data path
# path_to_depth = '/home/chna1572/workspace/depth_estimation/out/Datasets/nyu_depth_v2_labeled.mat'
#
#
#
# print (os.path.isfile(path_to_depth))
#
#
# #f = tables.open_file(path_to_depth)
#
#
# # read mat file
# f = h5py.File(path_to_depth)
#
#
#
#
# # read 0-th image. original format is [3 x 640 x 480], uint8
# img = f['images'][0]
#
#
# print ("Shape img", np.shape(img))
#
#
# # reshape
# img_ = np.empty([480, 640, 3])
# img_[:,:,0] = img[0,:,:].T
# img_[:,:,1] = img[1,:,:].T
# img_[:,:,2] = img[2,:,:].T
# plt.imshow(img_/255.0)
# plt.show()
#
# # imshow
# img__ = img_.astype('float32')
# # io.imshow(img__/255.0)
# # io.show()
#
#
#
# # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
# depth = f['depths'][0]
#
# # reshape for imshow
# depth_ = np.empty([480, 640, 3])
# depth_[:,:,0] = depth[:,:].T
# depth_[:,:,1] = depth[:,:].T
# depth_[:,:,2] = depth[:,:].T
#
# # io.imshow(depth_/4.0)
# # io.show()
#



a1 =  np.full(shape=(4, 360,640), fill_value=1)
print(a1.shape)

a2 = np.full(shape=(4,4), fill_value=2)
print(a2.shape)
print(np.ndim(a2))
out = np.dot(a2,a1)
# out = np.dot(a1,a2)
print(out.shape)