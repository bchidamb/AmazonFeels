import numpy as np
import pandas as pd
import os
import csv
from time import strftime
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

#from keras import backend as K
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
#from keras.wrappers.scikit_learn import KerasClassifier


def load_data(filename):

    return np.loadtxt(filename, skiprows=1, delimiter=' ')

def save_predictions(X, model):

    filename = 'svm_classifier_' + strftime('%b%d%H%M%S') + '.csv'
    preds = model.predict(X)
    print('preds:', preds)
    print('len(X)', X.shape[0])    
    preds = model.predict(X) #.reshape((len(X), 1)) # can't reshape for some reason
    preds = np.asarray(preds)    
    ids = (np.arange(1, X.shape[0] + 1))#.reshape((len(X), 1))

    inputArray = [[ids[i],preds[i]] for i in range(len(ids))]

    np.savetxt(
        os.path.join('predictions', filename),
        #np.hstack((ids, preds)),    # try vertical stack 
        inputArray,        
        fmt='%d',
        delimiter=',',
        header='Id,Prediction',
        comments=''
    )


def train(X, y, args={}):

    model = SVC(**args)
    model.fit(X, y)
    
    return model

def test(X, y, model):

    pred = np.reshape(model.predict(X), np.shape(y))
    return np.sum(pred == y) / len(y)

# Main function ===============================================================
# Load the training data
train_raw = load_data('training_data.txt')
train_raw = np.random.permutation(train_raw)

trans = TfidfTransformer() # our transformer

# Break the training data up into x and y components
X_train, y_train = train_raw[:, 1:], train_raw[:, 0]
n_features = len(X_train[0])

#X_train = trans.fit_transform(X_train)
trans.fit(X_train)
#X_train = trans.transform(X_train)

n_train = 18000
n_val = len(train_raw) - n_train
rx_train, ry_train = X_train[:n_train], y_train[:n_train]
X_val, y_val = X_train[n_train:], y_train[n_train:]

rx_train = trans.transform(rx_train)
X_val = trans.transform(X_val)

print(len(train_raw))
print('X_val', type(rx_train))

model = train(rx_train, ry_train, args={'kernel': 'linear'})

print('train / val split : %d / %d' % (n_train, n_val))
print('train acc :', test(rx_train, ry_train, model))
print('val acc :', test(X_val, y_val, model))


# Save the predictions
test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]
X_test = trans.transform(X_test)

save_predictions(X_test, model)

'''
<output>
train / val split : 19000 / 1000
train acc : 0.994157894737
val acc : 0.847
'''



