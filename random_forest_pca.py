import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
import os
from time import strftime

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
    
    
def decompose(X, d, args={}):
    
    pca_model = LatentDirichletAllocation(n_components=d, **args)
    pca_model.fit(X)
    
    return pca_model
    

def train(X, y, args={}):

    model = RandomForestClassifier(**args)
    model.fit(X, y)
    
    return model


def test(X, y, model):
    
    return np.sum(model.predict(X) == y) / len(y)


train_raw = load_data('training_data.txt')
n_train = 10000
n_val = len(train_raw) - n_train

X_train, y_train = train_raw[:, 1:][:n_train], train_raw[:, 0][:n_train]
X_val, y_val = train_raw[:, 1:][n_train:], train_raw[:, 0][n_train:]

# reduce dimensions from 1000 to 200
pca_model = decompose(X_train, 10)
X_train_red = pca_model.transform(X_train)
X_val_red = pca_model.transform(X_val)

model = train(X_train_red, y_train, args={})

print('train / val split : %d / %d' % (n_train, n_val))
print('train acc :', test(X_train_red, y_train, model))
print('val acc :', test(X_val_red, y_val, model))

test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]
X_test_red = pca_model.transform(X_test)

# save_predictions(X_test_red, model)

'''
<output>
train / val split : 10000 / 10000
train acc : 0.9892
val acc : 0.6557
'''

