import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Python 3.5

def load_data(filename):

    return np.loadtxt(os.path.join(path, filename), skiprows=1, delimiter=' ')


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

model = train(X_train, y_train)

print('train / val split : %d / %d' % (n_train, n_val))
print('train acc :', test(X_train, y_train, model))
print('val acc :', test(X_val, y_val, model))

'''
<output>
train / val split : 10000 / 10000
train acc : 0.9913
val acc : 0.7754
'''