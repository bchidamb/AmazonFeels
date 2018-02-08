import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

# Python 3.5

def load_data(filename):
    return np.loadtxt(filename, skiprows=1, delimiter=' ')



#model = Sequential()
#model.add(Dense(12, input_dim



train_raw = load_data('training_data.txt')
n_train = 10000
n_val = len(train_raw) - n_train

X_train, y_train = train_raw[:, 1:][:n_train], train_raw[:,0][:n_train]
X_val, y_val = train_raw[:, 1:][n_train:], train_raw[:, 0][n_train:]

print('shapes X_train:', X_train.shape)
print('shapes X_test:', X_val.shape)


model = Sequential()
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmxprop', loss='
