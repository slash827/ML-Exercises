import time
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras import backend as K
from ex3 import read_info

train_x, train_y, test_x = read_info()

print(train_x.shape)
print(test_x.shape)

# train_x = train_x.reshape(-1, 784)
# test_x = test_x.reshape(-1, 784)

test_y = np.loadtxt('test_y_100_percent_backup', dtype='uint8')

model = Sequential()
model.add(Dense(150, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Convert y_train into one-hot format
temp = []
for i in range(len(train_y)):
    temp.append(to_categorical(train_y[i], num_classes=10))
train_y = np.array(temp)

# Convert y_test into one-hot format
temp = []
for i in range(len(test_y)):
    temp.append(to_categorical(test_y[i], num_classes=10))
test_y = np.array(temp)


def nll1(y_true, y_pred):
    """ Negative log likelihood. """
    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


# model.summary()
model.compile(loss=nll1,
              optimizer='sgd',
              metrics=['acc'])

model.fit(train_x, train_y, epochs=10,
          validation_data=(test_x, test_y))

predictions = model.predict(test_x)
predictions = np.argmax(predictions, axis=1)

test_y = np.argmax(test_y, axis=1)
print(classification_report(test_y, predictions))
