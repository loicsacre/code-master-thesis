import errno
import os

import numpy as np
from scipy.spatial.distance import cdist


def compare(a, b, distance="cos"):
    """ Compute element-wise cosine similarity or euclidean distance of a and b
        a.shape = (m,f)
        b.shape = (n,f)

        Return c (with c.shape = (m, n))
    """
    if distance == "cos":
        # - (cdist(a, b, 'cosine') - 1) : works too but much slower
        # see https://stackoverflow.com/questions/56939272/what-is-the-best-way-to-implement-an-element-wise-cosine-similarity-in-python/56940543#56940543
        x = np.atleast_2d(np.sqrt(np.sum(a*a, axis=1))).T
        y = np.atleast_2d(np.sqrt(np.sum(b*b, axis=1))).T
        return a.dot(b.T) / x.dot(y.T)
    elif distance == "eucl":
        return cdist(a, b, "euclidean")
    elif distance == "eucl-norm":
        a = l2_normalize(a)
        b = l2_normalize(b)
        return cdist(a, b, "euclidean")


def l2_normalize(a):
    norm = np.sqrt(np.sum(a ** 2))
    return a/norm


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def mkdir(dirname):
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
