import numpy as np
import pandas as pd
import os
from time import strftime
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


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
def keras_model(layer1, dropout, opt):
	# create model
    model = Sequential()
    model.add(Dense(layer1, input_dim=1000))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def parameter_search(model, params, cv=None):

    return GridSearchCV(model, params, cv=cv)
    
#####
### Machine learning thingy goes here

# Load the training data
train_raw = load_data('training_data.txt')

# Break the training data up into x and y components
X_train, y_train = train_raw[:, 1:], train_raw[:, 0]

n_train = 18000
n_val = len(train_raw) - n_train
rx_train, ry_train = X_train[:n_train], y_train[:n_train]
X_val, y_val = X_train[n_train:], y_train[n_train:]

params = {
    'layer1': [50], # optimal value is 50
    'dropout': [0.7], # optimal value is 0.7
    'opt': ['rmsprop'], # optimal value is 'rmsprop'
    'epochs': [20], # optimal value is 20
    'batch_size': [1000] # optimal value is 1000
}

model = KerasClassifier(build_fn=keras_model, verbose=0)
search = parameter_search(model, params)

grid_result = search.fit(rx_train, ry_train)

with open('results.txt', 'w') as f:
    f.write('best score: ' + str(grid_result.best_score_) + '\n')
    f.write('best params: ' + str(grid_result.best_params_) + '\n')

print('best score:',grid_result.best_score_)
print('best params:',grid_result.best_params_)


'''
print('train / val split : %d / %d' % (n_train, n_val))
print('train acc :', test(rx_train, ry_train, model))
print('val acc :', test(X_val, y_val, model))


# Save the predictions
test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]

# save_predictions(X_test, model)


<output>
train / val split : 19000 / 1000
train acc : 0.994157894737
val acc : 0.847

'''
K.clear_session()