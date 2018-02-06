import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import os
from time import strftime

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

# Python 3.5

def load_data(filename):

    return np.loadtxt(filename, skiprows=1, delimiter=' ')


def save_predictions(X, model):

    filename = 'random_forest_' + strftime('%b%d%H%M%S') + '.csv'
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



def cross_val_train(X,y, n_fold, args={}):
     # Using 10-fold cross validation on 10,000 reviews
     # average the weights
     # Divide the data set into training and test sets

    min_val_err = 1
    min_val_model = 0 # dummy

    kf = KFold(n_splits=n_fold)
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #startModel = RandomForestClassifier(max_depth = 20)
        aModel = DecisionTreeClassifier(max_depth = 20)
        #aModel = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
        aModel.fit(X_train, y_train)

        # Pick the model with the lowest cross validation error
        val_err = test(X_test, y_test, aModel)
        if  val_err < min_val_err:
            print("Current cross val err:", val_err)
            min_val_model = aModel
            min_val_err = val_err

    return min_val_model

def train(X, y, args={}):

    #model = RandomForestClassifier(**args)
    model = RandomForestClassifier()
    model.fit(X, y)

    return model


def test(X, y, model):

    return np.sum(model.predict(X) == y) / len(y)


train_raw = load_data('training_data.txt')
n_train = 19000
n_val = len(train_raw) - n_train
n_folds = 5

X_train, y_train = train_raw[:, 1:][:n_train], train_raw[:, 0][:n_train]
X_val, y_val = train_raw[:, 1:][n_train:], train_raw[:, 0][n_train:]

#model = cross_val_train(X_train, y_train, n_folds, args={})
model = train(X_train, y_train, args={})

print('train / val split : %d / %d' % (n_train, n_val))
print('train acc :', test(X_train, y_train, model))
print('val acc :', test(X_val, y_val, model))

test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]

save_predictions(X_test, model)

'''
<output>
train / val split : 10000 / 10000
train acc : 0.9913
val acc : 0.7754
'''
