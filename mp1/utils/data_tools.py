"""
Implements feature extraction and other data processing helpers.
"""

import numpy as np
import skimage
from skimage import filters
from numpy import fft


def preprocess_data(data, process_method='default'):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1]
          2. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Apply laplacian filter with window size of 11x11. (use skimage)
          3. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'default':
        img = data['image']/255
        img = np.stack([skimage.filters.laplace(img[idx,:,:], ksize=11) for idx in range(np.size(data['image'],0))])
        data['image'] = img
        # img_mean = np.mean(img, axis=0)
        # img_mean = np.concatenate( [img_mean[None,...] for i in range(np.size(img,0))], axis=0 )
        # img = img-img_mean
        data = remove_data_mean(data)
        data['image'] = np.stack( [ data['image'][idx,:,:].flatten() for idx in range(np.size(data['image'],0)) ] )
    elif process_method == 'raw':
        data['image'] = data['image']/255
        # img_mean = np.mean(img, axis=0)
        # img_mean = np.concatenate( [img_mean[None,...] for i in range(np.size(img,0))], axis=0 )
        # img = img-img_mean
        data = remove_data_mean(data)
        data['image'] = np.stack( [ data['image'][idx,:,:].flatten() for idx in range(np.size(data['image'],0)) ] )
    elif process_method == 'custom':
        pass

    return data


def compute_image_mean(data):
    """ Computes mean image.
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """
    image_mean = np.mean(data['image'], axis=0)
    # image_mean = None
    return image_mean


def remove_data_mean(data):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    # img = data['image']
    img_mean = compute_image_mean(data)
    data['image'] = np.stack( [ (data['image'][idx,:,:] - img_mean) for idx in range(np.size(data['image'],0)) ] )
    return data
