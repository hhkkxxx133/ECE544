"""Input and output helpers to load in data.
"""

import pickle
import numpy as np
from numpy import genfromtxt


def read_dataset(input_file_path):
    """Read input file in csv format from file.
    In this csv, each row is an example, stored in the following format.
    label, pixel1, pixel2, pixel3...pixel748.

    Args:
        input_file_path(str): Path to the csv file.
    Returns:
        (1) feature (np.ndarray): Array of dimension (N, ndims) containing the
        images scaled to the range of [0,1].
        (2) label (np.ndarray): Array of dimension (N,) containing the label.
    """
    data = genfromtxt(input_file_path, delimiter=',')
    label = data[:,0]
    feature = data[:,1:]/255.0
    return feature, label
