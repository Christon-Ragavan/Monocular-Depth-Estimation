

import numpy as np
import os
import glob
import cv2

import matplotlib.pyplot as plt


# def


def im_show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

dataset_location = "/home/chna1572/workspace/depth_estimation/nyu_datasets"
test_pic = "/home/chna1572/workspace/depth_estimation/nyu_datasets/00000.png"
print (os.path.isdir(dataset_location))

img = cv2.imread(test_pic,1)

# im_show(img)

print (np.shape(img))
