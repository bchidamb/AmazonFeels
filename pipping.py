import numpy as np
import pandas as pd
import os
import csv
from time import strftime
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

def load_data(filename):

    return np.loadtxt(filename, skiprows=1, delimiter=' ')
    
def save_predictions(X, model):

    filename = 'pipe' + strftime('%b%d%H%M%S') + '.csv'
    preds = model.predict(X)
    print('preds:', preds)
    print('len(X)', X.shape[0])    
    preds = model.predict(X)  # can't reshape for some reason
    preds = np.asarray(preds)    
    ids = (np.arange(1, X.shape[0] + 1))

    inputArray = [[ids[i], preds[i]] for i in range(len(ids))]

    np.savetxt(
        os.path.join('predictions', filename),
        inputArray,        
        fmt='%d',
        delimiter=',',
        header='Id,Prediction',
        comments=''
    )
    
# Grid Search for finding optimal hyperparameters
def parameter_search(model, params, cv=None):

    return GridSearchCV(model, params, cv=cv)
    
# Pipeline chaining multiple estimators into one
def pipeline_estim(estimators):
    
    return Pipeline(estimators)
    
    
def test(X, y, model):

    pred = np.reshape(model.predict(X), np.shape(y))
    return np.sum(pred == y) / len(y)

    
    
    
    
# main =================

# Load the training data
train_raw = load_data('training_data.txt')

# Break the training data up into x and y components
X_train, y_train = train_raw[:, 1:], train_raw[:, 0]

n_train = 18000
n_val = len(train_raw) - n_train
rx_train, ry_train = X_train[:n_train], y_train[:n_train]
X_val, y_val = X_train[n_train:], y_train[n_train:]

estimators = [('reduce_dim', PCA()), ('Tfid', TfidfTransformer()), ('clf_SVM', SVC(kernel='linear'))]
pipe = pipeline_estim(estimators)

print('fitting...')
pipe_result = pipe.fit(rx_train, ry_train)
print('pipe result:', pipe_result)

print('train accuracy:', test(rx_train, ry_train, pipe))
print('valid accuracy:', test(X_val, y_val, pipe))

# Save the predictions
test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]
#X_test = trans.transform(X_test)

save_predictions(X_test, pipe)








