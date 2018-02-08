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


def save_predictions(X, model):

    filename = 'keras_classifier_' + strftime('%b%d%H%M%S') + '.csv'
    preds = model.predict(X).reshape((len(X), 1))
    ids = (np.arange(1, len(X) + 1)).reshape((len(X), 1))

    np.savetxt(
        os.path.join('predictions', filename),
        np.hstack((ids, preds)),
        fmt='%d',
        delimiter=',',
        header='Id,Prediction',
        comments=''
    )


def test(X, y, model):

    pred = np.reshape(model.predict(X), np.shape(y))
    return np.sum(pred == y) / len(y)
    




## Model creation goes here
def keras_model():
	# create model
    model = Sequential()
    model.add(Dense(50, input_dim=n_features))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# normalize the input, from set 2
# if None passed into means, stds calcucate it
def normalize(X, means, stds):
    X_trans = np.copy(X)
    X_trans = X_trans.T # transpose the columns and rows, normalizing rows
    # go through each column and find the mean and standard deviation
    if means is None:
        print('means is None, calculating means/stds...')
        means = [1]
        stds = [1]
        for a in range(1, len(X_trans)):
            means.append(np.average(X_trans[a]))
            stds.append(np.std(X_trans[a]))
    
    # go through the rows and columns of X and update each element to normalize it
    for i in range(1, len(X_trans)):
        for j in range(len(X_trans[0])):
            X_trans[i][j] = (X_trans[i][j] - means[i]) / stds[i]
    
    return [X_trans.T, means, stds]

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

print(len(train_raw))
print('X_val', type(rx_train), rx_train)

rx_train, means, stds = normalize(rx_train, None, None)
X_val, meansTemp, stdsTemp = normalize(X_val, means, stds)

model = KerasClassifier(build_fn=keras_model, epochs=20, batch_size=1000, verbose=1)
model.fit(rx_train, ry_train)

print('train / val split : %d / %d' % (n_train, n_val))
print('train acc :', test(rx_train, ry_train, model))
print('val acc :', test(X_val, y_val, model))


# Save the predictions
test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]
X_test, meansTemp, stdsTemp = normalize(X_test, means, stds)

save_predictions(X_test, model)

'''
<output>
train / val split : 19000 / 1000
train acc : 0.994157894737
val acc : 0.847
'''

K.clear_session()
