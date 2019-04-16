import keras as keras
from tqdm import tqdm
import itertools
import numpy as np


def pairwise(itt):
    a, b = itertools.tee(itt)
    next(b, None)
    return zip(a, b)


def hash_data(data):
    return np.array(list(map(lambda x: hash(str(x)), data)))


def bin_array(array, low=-1, high=1, bins=10):
    e = (high - low) / bins
    return ((array - low) / e).astype(int)


# addding noise is necessary to prevent infinite MI (i.e prevents division by zero for some MI estimators)
def add_noise(data, mean=0, std=0.01):
    return data + np.random.normal(mean, std, data.shape)


def __add_noise_value(n, noise_function):
    return n + noise_function()


class ProgressBarCallback(keras.callbacks.Callback):
    """
    Calls informationProcessor to calculate mutual information per batch
    """
    def __init__(self, no_of_batches):
        super().__init__()
        self.__progress = tqdm(total=no_of_batches)

    def on_batch_end(self, batch, logs=None):
        self.__progress.update(1)

    def on_train_end(self, logs=None):
        self.__progress.close()
