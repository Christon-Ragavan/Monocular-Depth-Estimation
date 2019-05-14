
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.utils import np_utils
import extraction_tools
import numpy as np
import matplotlib.pyplot as plt

def test():
    print("Test")




def UnetPredict (model, raw_test, depth_test):
    predictions = model.predict(raw_test)

    # print("Predictions [0] :", predictions[0])
    print(" ------------#######\nPredictions [0] Shape::", np.shape(predictions[0]))

    # print("Predicted_outPut", np.argmax(predictions[0]))
    # print ("Truth", np.argmax(y_test_label[0]))

    op = extraction_tools.transformation.raw_image_tansformation(raw_test[0])
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


def evaluation_test(raw_train, raw_test, depth_train, depth_test):

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

