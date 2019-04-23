import keras.models as model

# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt


# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]





model_load = model.load_model("/home/chna1572/workspace/depth_estimation/scripts/model_02.h5")

test_loss, test_acc = model_load.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)
print('Test Loss:', test_loss)




predictions = model_load.predict(X_test)

# print("Predictions [0] :",predictions[0])
print("Predicted_outPut", np.argmax(predictions[0]))
print ("Truth", np.argmax(y_test[0]),y_test[0])
