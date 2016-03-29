import os
import numpy as np
import theano
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation
from keras.optimizers import SGD

def load_data():
    data = np.empty((42000,1,28,28),dtype="float32")
    label = np.empty((42000,),dtype="uint8")

    return data,label

WIDTH = 512
DEEP = 3
DROPOUT_PROB = 0.5
LR = 0.1
MOMENTUM = 0.9

if __name__ == '__main__':
    # build model
    model = Sequential()

    model.add(Dense(WIDTH, input_dim=1800, init='uniform'))
    model.add(Activation('sigmoid'))
    for i in xrange(DEEP-1):
        model.add(MaxoutDense(WIDTH, nb_feature=2, init='uniform'))
        model.add(Dropout(DROPOUT_PROB))
    model.add(MaxoutDense(WIDTH, nb_feature=2, init='uniform'))

    # TODO
    cos_sim = theano.function(inputs=[y_true, y_pred], outputs=l)
    sgd = SGD(lr=LR, decay=1e-6, momentum=MOMENTUM, nesterov=True)
    model.compile(loss=cos_sim, optimizer=sgd)

    # read data
    data, label = load_data()

    # training
    model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
    score = model.evaluate(X_test, y_test, batch_size=16)