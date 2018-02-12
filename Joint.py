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
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data(filename):

    return np.loadtxt(filename, skiprows=1, delimiter=' ')
    
def save_predictions(X, model):

    filename = 'Joint' + strftime('%b%d%H%M%S') + '.csv'
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
def parameter_search(model, params, numJobs, cv=None):

    return GridSearchCV(model, params, n_jobs=numJobs, cv=cv)
    
# Pipeline chaining multiple estimators into one
def pipeline_estim(estimators):
    
    return Pipeline(estimators)
    
    
def test(X, y, model):

    pred = np.reshape(model.predict(X), np.shape(y))
    return [np.sum(pred == y) / len(y), pred]
    
    
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
    
    
def jointPred(pred0, pred1, pred2):
    amalgPred = []
    for i in range(len(pred1)):
        # if 2/3 say yes, then yes
        if (pred0[i] + pred1[i] + pred2[i]) >= 2:
            amalgPred.append(1)
        else:
            amalgPred.append(0)
            
    return amalgPred
    
# main =================

# Load the training data
train_raw = load_data('training_data.txt')
train_raw = np.random.permutation(train_raw)

# Break the training data up into x and y components
X_train, y_train = train_raw[:, 1:], train_raw[:, 0]

n_train = 19000
n_val = len(train_raw) - n_train
rx_train, ry_train = X_train[:n_train], y_train[:n_train]
X_val, y_val = X_train[n_train:], y_train[n_train:]

# run pipe and fit it ---------------------------------------------------------
estimators = [('Tfid', TfidfTransformer()), ('clf_SVM', SVC(kernel='linear'))]
pipe = pipeline_estim(estimators)

print('fitting...')
pipe_result = pipe.fit(rx_train, ry_train)
#print('pipe result:', pipe_result)
#print('pipe params:', pipe.get_params())
trainAccSVM, predSVMTrain = test(rx_train, ry_train, pipe)
valAccSVM, predSVMVal = test(X_val, y_val, pipe)
print('train acc SVM:', trainAccSVM)
print('vali acc SVM:', valAccSVM)

# run keras nerual net and fit it ----------------------------------------------
params = {
    'layer1': [50], # optimal value is 50
    'dropout': [0.7], # optimal value is 0.7
    'opt': ['rmsprop'], # optimal value is 'rmsprop'
    'epochs': [20], # optimal value is 20
    'batch_size': [1000] # optimal value is 1000
}
modelNN = KerasClassifier(build_fn=keras_model, layer1=50, dropout=0.7, opt='rmsprop',     epochs=20, batch_size = 1000, verbose=0)
modelNN.fit(rx_train, ry_train)
trainAccNN, predNNTrain = test(rx_train, ry_train, modelNN)
valAccNN, predNNVal = test(X_val, y_val, modelNN)
print('train acc NN:', trainAccNN)
print('vali acc NN:', valAccNN)


# run random forest and fit it ------------------------------------------------
estimators1 = [('Tfid', TfidfTransformer()), ('clf_RF', RandomForestClassifier())]
pipeline_estim(estimators)
modelRF = pipeline_estim(estimators1)
modelRF.fit(rx_train, ry_train)
trainAccRF, predRFTrain = test(rx_train, ry_train, modelRF)
valAccRF, predRFVal = test(X_val, y_val, modelRF)
print('train acc RF:', trainAccRF)
print('vali acc RF:', valAccRF)


# try doing a joint prediction
overallPredTrain = jointPred(predSVMTrain, predNNTrain, predRFTrain)
overallPredVal   = jointPred(predSVMVal, predNNVal, predRFVal)

print('train acc joint', np.sum(overallPredTrain == ry_train) / len(ry_train))
print('val acc joint', np.sum(overallPredVal == y_val) / len(y_val))


# Save the predictions
test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]

predSVM = pipe.predict(X_test)
predNN  = modelNN.predict(X_test)
predRF  = modelRF.predict(X_test)

filename = 'Joint' + strftime('%b%d%H%M%S') + '.csv'
preds = jointPred(predSVM, predNN, predRF)
#print('preds:', preds)
print('len(X)', len(X_test))    
preds = np.asarray(preds)    
ids = (np.arange(1, len(X_test) + 1))

inputArray = [[ids[i], preds[i]] for i in range(len(ids))]

np.savetxt(
    os.path.join('predictions', filename),
    inputArray,        
    fmt='%d',
    delimiter=',',
    header='Id,Prediction',
    comments=''
)







