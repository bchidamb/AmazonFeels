import numpy as np
import pandas as pd
import os
from time import strftime
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier


def load_data(filename):

    return np.loadtxt(filename, skiprows=1, delimiter=' ')


## Model creation goes here
def keras_model():
	# create model
    model = Sequential()
    model.add(Dense(1, input_dim=n_features))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


#####
### Machine learning thingy goes here

# Load the training data
train_raw = load_data('training_data.txt')
train_raw = np.random.permutation(train_raw)

# Break the training data up into x and y components
X_train, y_train = train_raw[:, 1:], train_raw[:, 0]
n_features = len(X_train[0])

n_train = 18000
n_val = len(train_raw) - n_train
rx_train, ry_train = X_train[:n_train], y_train[:n_train]
X_val, y_val = X_train[n_train:], y_train[n_train:]

model = keras_model()
model.fit(rx_train, ry_train, epochs=20, batch_size=1000, verbose=0)

# visualization
dense1 = model.layers[0]

weights = list(np.reshape(dense1.get_weights()[0], (1000,)))
weights_sq = list(np.reshape(dense1.get_weights()[0], (1000,)) ** 2)

with open('training_data.txt') as f:
    words = f.readline().strip().split(' ')[1:]
    
paired = list(zip(weights, words))
paired = sorted(paired, reverse=True)

paired_sq = list(zip(weights_sq, words))
paired_sq = sorted(paired_sq)

print('Top 10: positively correlated')
for i, e in enumerate(paired[:10]):
    print(i + 1, e)

print('Top 10: negatively correlated')
for i, e in enumerate(paired[:-11:-1]):
    print(i + 1, e)
    
print('Top 10: least correlated')
for i, e in enumerate(paired_sq[:10]):
    print(i + 1, e)
    
K.clear_session()
