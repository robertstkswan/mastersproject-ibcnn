import numpy as np
import os
import sys
import scipy.io as sio
import keras
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

supported_data_sets = ["Tishby", "MNIST"]
__SEED__ = 2424


def load_data(data_set, train_size):
    if data_set == 'MNIST':
        return get_mnist(train_size)
    elif data_set == "Tishby":
        return get_tishby(train_size)
    elif data_set == "TEST":
        train, test, cat = get_mnist(train_size)
        train = train[0][:1000], train[1][:1000]
        test = test[0][:1000], test[1][:1000]
        return train, test, cat
    else:
        raise ValueError("Data set {} is not supported, supported data sets: {}"
                         .format(data_set, supported_data_sets))


def get_mnist(train_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = np.concatenate((x_train, x_test))
    labels = np.concatenate((y_train, y_test))

    data = data.reshape(data.shape[0], 28 * 28)
    data = data.astype('float32') / 255
    labels = keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, random_state=__SEED__)
    return (x_train, y_train), (x_test, y_test), 10


def get_tishby(train_size):
    name = "data/var_u"
    d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
    data = d['F']
    y = d['y']
    labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_size, random_state=__SEED__)
    return (x_train, y_train), (x_test, y_test), 2


def __random(dim):
    m = np.random.random_integers(0, dim)
    arr = np.ones(dim)
    arr[:m] = 0
    np.random.shuffle(arr)
    return arr


def parameters_data(parser):
    parameters = parser.add_argument_group('Data Set parameters')

    parameters.add_argument('--data_set',
                        '-ds', dest='data_set', default='Tishby',
                        help='choose a data set, available: {}'.format(supported_data_sets) +
                             ', Tishby - data set used by Tishby in the original paper')

    parameters.add_argument('--train_size',
                        '-ts', dest='train_size', default=0.8,
                        type=float, help='Training size')

