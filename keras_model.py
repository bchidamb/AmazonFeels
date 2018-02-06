import numpy as np
import pandas as pd
import os
from time import strftime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

# Load the training data
train_raw = load_data('training_data.txt')
X_train, y_train = train_raw[:, 1:], train_raw[:, 0]
n_features = len(X_train[0])


## Model creation goes here
def keras_model():
	# create model
    model = Sequential()
    model.add(Dense(100, input_dim=1000,
                    kernel_initializer='normal',
                    activation='relu'))
                    #kernel_regularizer=regularizers.l2(0.01),
                    #activity_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))

    # Compile model
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#####
### Machine learning thingy goes here

# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=keras_model, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
# print("Keras Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model = KerasClassifier(build_fn=keras_model, epochs=10, batch_size=100, verbose=0)
model.fit(X_train, y_train)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print("Average cross validation error: ", results.mean())


#####

# Save the predictions
test_raw = load_data('test_data.txt')
X_test = test_raw[:, :]

save_predictions(X_test, model)

'''
<output>
train / val split : 10000 / 10000
train acc : 0.9913
val acc : 0.7754
'''
