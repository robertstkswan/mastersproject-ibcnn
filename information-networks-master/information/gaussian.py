import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy.spatial as ss
import scipy.optimize as so


def local_likelihood(data, k):
    """
    :param data: set of all points
    :param k: nearest neighbour
    :return: a function (x, mean, L) -> eq 29.
    """
    tree = ss.cKDTree(data)
    H = np.identity(len(data[0]))

    def calculation(x, mean, L):
        """
        :param x: a single point
        :param mean: mean
        :param L: Sigma = L.transpose() * L
        :return: eq 29.
        """
        L = np.matrix(L)
        sigma = L.transpose() * L
        _, k_nearest = tree.query(x, k=k+1)
        res = 0
        for neighbour in k_nearest:
            res += N(neighbour, x, H) * math.log2(N(neighbour, mean, sigma))
        res /= k+1
        res = -N(x, mean, sigma + H)
        return res

    return calculation


def N(x, mean, covar):
    return multivariate_normal.pdf(x, mean=mean, cov=covar)


def entropy(data):
    h = 0
    for x in data:
        eq29 = local_likelihood(data, 4)
        mean, sig = np.zeros(len(x)), np.zeros(len(x))

        eq29(data[0], mean, sig)

        sig = np.matrix(sig).transpose() * np.matrix(sig)
        sig = np.identity(len(x))
        f = N(data, mean, sig)
        h = h - (math.log2(f))/len(data)
    return h


def calculate_information_data(data_x, data_y):
    x = data_x
    y = data_y

    h_x = entropy(x)
    h_y = entropy(y)
    h_x_y = entropy(zip(x, y))

    mutual_information = h_x + h_y - h_x_y
    return mutual_information
