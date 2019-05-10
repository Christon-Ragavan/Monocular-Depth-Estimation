
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
import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
# import work


def de_conv():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(3,480, 640), activation='relu'))
    model.add(Conv2D(15, (3, 3), activation='relu'))

    model.add(Conv2DTranspose(15, (3,3), activation='relu'))
    model.add(Conv2DTranspose(1, (5,5), activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model





def train(raw_train, raw_test, depth_train, depth_test):
    print("building model...")
    model = de_conv()

    # # Fit the model
    model.fit(raw_train, depth_train, validation_data=(raw_test, depth_test), epochs=1)
    model.save('/home/chna1572/workspace/depth_estimation/scripts/first_u_net_model.h5')

    test_loss, test_acc = model.evaluate(raw_test, depth_test)

    print('Test accuracy:', test_acc)
    print('Test Loss:', test_loss)

    predictions = model.predict(raw_test)

    # print("Predictions [0] :", predictions[0])
    print(" ------------#######\nPredictions [0] Shape::", np.shape(predictions[0]))

    # print("Predicted_outPut", np.argmax(predictions[0]))
    # print ("Truth", np.argmax(y_test_label[0]))

    op = utils.transformation.raw_image_tansformation(raw_test[0])
    print(np.shape(op))

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(op.reshape(480, 640, 3))
    ax2.imshow(depth_test[0].reshape(480, 640))
    ax3.imshow(predictions[0].reshape(480, 640))

    ax1.title.set_text('Original_RGB_image')
    ax2.title.set_text('Ground_truth_Depth_map')
    ax3.title.set_text('Predicted_Depth_map')

    plt.show()

def import_data():
    trans_raw_images, trans_depth_images = get_dataset.run()
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

    raw_train = raw_train / 255.0
    raw_test = raw_test / 255.0
    depth_train = depth_train / 255.0
    depth_test = depth_test / 255.0

    print("RAW", raw_train.shape, raw_test.shape)
    # depth_train = raw_train
    # depth_test = raw_test


    # fig = plt.figure()
    # a1 = fig.add_subplot(131)
    # a2 = fig.add_subplot(132)
    # a3 = fig.add_subplot(133)
    #
    # a1.imshow(op.reshape(480, 640, 3))
    # a2.imshow(depth_test[0].reshape(480, 640))
    # a3.imshow(op.reshape(480, 640, 3))
    #
    # a1.title.set_text('Original_RGB_image')
    # a2.title.set_text('Ground_truth_Depth_map')
    # a3.title.set_text('raw_train')

    plt.show()

    # np.savez('features.npz',  raw_train= raw_train,  raw_test =raw_test, depth_train=depth_train, depth_test=depth_test)

    return raw_train, raw_test, depth_train, depth_test


def run():
    K.set_image_dim_ordering('th')
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    raw_train, raw_test, depth_train, depth_test = import_data()
    train(raw_train, raw_test, depth_train, depth_test)


    pass


def evaluation(raw_train, raw_test, depth_train, depth_test):

    model = tf.keras.models.load_model('/home/chna1572/workspace/depth_estimation/scripts/first_u_net_model.h5')

    print(model.keys())
    # model.save('/home/chna1572/workspace/depth_estimation/scripts/first_u_net_model.h5')

    # test_loss, test_acc = model.evaluate(raw_test, depth_test)

    # print('Test accuracy:', test_acc)
    # print('Test Loss:', test_loss)

    predictions = model.predict(raw_test)

    # print("Predictions [0] :", predictions[0])
    print("Predictions [0] Shape::", np.shape(predictions[0]))

    # print("Predicted_outPut", np.argmax(predictions[0]))
    # print ("Truth", np.argmax(y_test_label[0]))

    op = utils.transformation.raw_image_tansformation(raw_test[0])
    print(np.shape(op))

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(op.reshape(480, 640, 3))
    ax2.imshow(depth_test[0].reshape(480, 640))
    ax2.imshow(predictions[0].reshape(480, 640))

    ax1.title.set_text('Original_RGB_image')
    ax2.title.set_text('Ground_truth_Depth_map')
    ax3.title.set_text('Predicted_Depth_map')

def run_evaluation():
    raw_train, raw_test, depth_train, depth_test = import_data()
    evaluation(raw_train, raw_test, depth_train, depth_test)



if __name__ =="__main__":
    run()
    # run_evaluation()