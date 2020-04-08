from scipy.io import loadmat
import numpy as np


def to_numpy(data):
    features = data["fts"]
    labels = data["labels"]
    data_matrix = np.concatenate((features, labels), axis=1)
    return data_matrix


def read(path):
    data = loadmat(path)
    data = to_numpy(data)
    return data



