
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


import eval
import models
import get_dataset
import extraction_tools
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
K.set_image_dim_ordering('th')




def train(raw_train, raw_test, depth_train, depth_test):
    batch_size = 64
    print("building model...")
    model = models.create_model_v1()
    # model = models.u_net()
    gegabytes = extraction_tools.utils.get_model_memory_usage(batch_size, model)
    print("\n\n\n\n----------------- ALLOCATING ",gegabytes,"GB ------------------------------\n\n\n\n")


    # # Fit the model
    model.fit(raw_train, depth_train, validation_data=(raw_test, depth_test), batch_size=batch_size, epochs=1)
    # model.save('/home/chna1572/workspace/depth_estimation/scripts/first_u_net_model.h5')
    print("Done Training..")


    test_loss, test_acc = model.evaluate(raw_test, depth_test)

    print('Test accuracy:', test_acc)
    print('Test Loss:', test_loss)

    return model





if __name__ =="__main__":
    path = '/home/chna1572/workspace/depth_estimation/out/Datasets/nyu_depth_v2_labeled.mat'

    raw_train, raw_test, depth_train, depth_test = get_dataset.import_nyu_sample_data(do_extract = False ,path_to_depth=path)

    model = train(raw_train, raw_test, depth_train, depth_test)

    eval.UnetPredict(model, raw_test, depth_test)
