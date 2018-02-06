import numpy as np
import pandas as pd
import os
from time import strftime
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import LatentDirichletAllocation


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


def decompose(X, d, args={}):
    
    pca_model = LatentDirichletAllocation(n_components=d, **args)
    pca_model.fit(X)
    
    return pca_model


# Load the training data
train_raw = load_data('training_data.txt')
n_train = 19000
n_val = len(train_raw) - n_train

# Break the training data up into x and y components
X_train, y_train = train_raw[:, 1:][:n_train], train_raw[:, 0][:n_train]
X_val, y_val = train_raw[:, 1:][n_train:], train_raw[:, 0][n_train:]

# reduce dimensions from 1000 to n_features
n_features = 500
pca_model = decompose(X_train, n_features)
X_train_red = pca_model.transform(X_train)
X_val_red = pca_model.transform(X_val)


## Model creation goes here
def keras_model():
	# create model
    model = Sequential()
    model.add(Dense(150, input_dim=n_features))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#####
### Machine learning thingy goes here

model = KerasClassifier(build_fn=keras_model, epochs=10, batch_size=100, verbose=1)
model.fit(X_train_red, y_train)

print('train / val split : %d / %d' % (n_train, n_val))
print('train acc :', test(X_train_red, y_train, model))
print('val acc :', test(X_val_red, y_val, model))


# Save the predictions
test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]

# Perhaps cluster the test data

# save_predictions(X_test, model)

'''
<output>
train / val split : 10000 / 10000
train acc : 0.9913
val acc : 0.7754
'''

K.clear_session()