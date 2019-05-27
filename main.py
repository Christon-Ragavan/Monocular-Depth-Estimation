
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import datetime
import eval
import models
import get_dataset
import extraction_tools
from keras import backend as K

from time import gmtime, strftime
K.tensorflow_backend._get_available_gpus()
# K.set_image_dim_ordering('th')




def train(input_shape, raw_train, raw_test, depth_train, depth_test, epoc):
    batch_size = 64
    print("building model...")
    model = models.u_net(input_shape)
    # model = models.u_net()
    gegabytes = extraction_tools.utils.get_model_memory_usage(batch_size, model)
    print("\n\n\n\n----------------- ALLOCATING ",gegabytes,"GB ------------------------------\n\n\n\n")


    # # Fit the model
    model.fit(raw_train, depth_train, validation_data=(raw_test, depth_test), batch_size=batch_size, epochs=epoc)
    # model.save('/home/chna1572/workspace/depth_estimation/scripts/first_u_net_model.h5')
    print("Done Training..")


    test_loss, test_acc = model.evaluate(raw_test, depth_test)

    print('Test accuracy:', test_acc)
    print('Test Loss:', test_loss)

    return model


def _get_MNIST_data():
    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')



    raw_train = x_train.copy()
    raw_test = x_test.copy()
    depth_train = x_train.copy()
    depth_test = x_test.copy()
    print(raw_train.shape, raw_test.shape)

    return raw_train, raw_test, depth_train, depth_test


if __name__ =="__main__":

    do_save_model = False
    do_extract = False
    do_save_features = False
    do_crop = True
    do_train = True
    epoc = 2
    # max_file_extract =

    path = '/home/chna1572/workspace/depth_estimation/out/Datasets/nyu_depth_v2_labeled.mat'

    path_nyu_dir = '/home/chna1572/workspace/depth_estimation/xtract/Datasets/nyu_datasets'

    raw_train, raw_test, depth_train, depth_test = get_dataset.import_nyu_sample_data(do_extract = do_extract ,
                                                                                      crop = do_crop,
                                                                                      do_save_features = do_save_features,
                                                                                      path_to_depth=path)



    raw_train, raw_test, depth_train, depth_test = get_dataset.import_nyu_dir(do_extract = do_extract ,crop = do_crop,do_save_features = do_save_features,path_to_dir=path_nyu_dir)



    print("\n\n\nALL SHAPES \n\n\n",np.shape(raw_train), np.shape(raw_test), np.shape(depth_train), np.shape(depth_test))
    input_shape = (raw_train.shape[1], raw_train.shape[2], raw_train.shape[3])
    print("####### INPUT SHAPE ########## ---->>>", input_shape)


    if do_train:
        # raw_train = raw_train[:100]
        # raw_test = raw_test[:25]
        # depth_train = depth_train[:100]
        # depth_test = depth_test[:25]
        input_shape = (raw_train.shape[1], raw_train.shape[2], raw_train.shape[3])
        print("####### INPUT SHAPE ########## ---->>>",input_shape)
        model = train(input_shape, raw_train, raw_test, depth_train, depth_test, epoc)

        if do_save_model:
            loc = '/home/chna1572/workspace/depth_estimation/scripts/saved_models/model_unet' + strftime("%d_%m_%Y-%H_%M_%S", gmtime())

            model_json = model.to_json()
            with open(loc+'.json', "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(loc+"_unet.h5")
            print("Saved model to disk")


        eval.UnetPredict_loaded_fld(model, raw_test, depth_test)



