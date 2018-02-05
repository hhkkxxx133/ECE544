"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.
    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.
    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,28,28)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """

    f = open(data_txt_file, 'r')

    data = {}
    raw_data = f.readlines()
    img_data = np.zeros((len(raw_data), 28, 28))
    label_data = []
    idx = 0
    for line in raw_data:
        file_name, label = line.split('\t')
        file_path = os.path.join(image_data_path, file_name)
        img = io.imread(file_path)
        # img = np.expand_dims(img, axis=0)
        # img_data = np.concatenate((img_data, img), axis=0)
        img_data[idx,:,:] = img
        label_data.append(int(label)) 
        idx += 1

    data['image'] = img_data
    data['label'] = np.asarray(label_data)

    f.close()
    # print ("done")
    return data


def write_dataset(data_txt_file, data):
    """Write python dictionary data into csv format for kaggle.
    Args:
        data_txt_file(str): path to the data txt file.
        data(dict): A Python dictionary with keys 'image' and 'label',
          (see descriptions above).
    """
    with open('result.csv','w') as f:
        f.write('Id,Prediction\n')
        for idx in range(data['label'].shape[0]):
            f.write('test_'+str(idx).zfill(5)+'.png,'+str(data['label'].flatten()[idx])+'\n')
    # pass

