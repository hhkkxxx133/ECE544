# dataset.py
# Created by dwang49
# This files implements the dataset class similar to the tf.MNIST
# dataset class, where the samples are preloaded into the memory.
# Shuffled and randomly sampled to provide minibatch at runtime.


from multiprocessing import Pool
import numpy as np
import os
from scipy.io import loadmat


from config import workspace_config
from feature import feature_generator


# Construct a dataset
class DataSet():

    # Sanity check and logging
    def __init__(self, key='train'):

        # Specify data path
        assert key in ('train', 'val', 'test')
        if key == 'train':
            self._data_path = workspace_config.train_path
        elif key == 'val':
            self._data_path = workspace_config.val_path
        elif key == 'test':
            self._data_path = workspace_config.test_path
        else:
            print('No magic here!')

        # Specify label mapping
        self._label_map = {
            'yes': 0,
            'no': 1,
            'up': 2,
            'down': 3,
            'left': 4,
            'right': 5,
            'on': 6,
            'off': 7,
            'stop': 8,
            'go': 9,
            'silence': 10,
            'unknown': 11,
        }
        self._class_num = 12

        # Fetch feature generator
        self._fgen = feature_generator()

        # Do the actuall work
        self._data, self._label = self.load_data()

        # Marker
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = self._data.shape[0]

        return

    # tf mnist flavor next_batch fetcher
    def next_batch(self, batch_size, shuffle=True):

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle_data()
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples, :, :]
            label_rest_part = self._label[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                self.shuffle_data()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end, :, :]
            label_new_part = self._label[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end, :, :], self._label[start:end]

    # Data Shuffler
    # Shuffle entire data and label consistently
    def shuffle_data(self):

        assert self._data.shape[0] == self._label.shape[0]

        p = np.random.permutation(self._data.shape[0])

        self._data = self._data[p]
        self._label = self._label[p]

        return

    # Data loader
    # Load all mat files from one directory with multithreading
    def load_data(self, nworkers=5):

        # Fetch all files in data dir
        files = os.listdir(self._data_path)

        # Logging
        print('\n--------------------------------------------------')
        print('Start proccessing {} files...'.format(len(files)))
        print('Data directory: {}'.format(self._data_path))
        print('Number of workers: {}'.format(nworkers))

        # Multiprocessing
        with Pool(nworkers) as p:
            raw_data = p.map(self.load_file, files)

        print('Finished processing all files!')
        print('--------------------------------------------------\n')

        # Concatenate and generate labels
        data = np.concatenate([f for (f, _, _) in raw_data], axis=0)
        label = self.gen_label([(l, n) for (_, l, n) in raw_data])
        del raw_data

        return data, label

    # Single file data loader
    # Load single mat file and extract features
    def load_file(self, file_name):

        # Logging
        print('+ Processing file: {}...'.format(file_name))

        # Open .mat file
        mat = loadmat('/'.join((self._data_path, file_name)))
        data, label = mat['data'], mat['label'][0]

        # Extract features
        feature = self._fgen.feature_extract(data)
        del data

        # Logging
        print('> Finished processing file: {}...'.format(file_name))

        # Pack into tuple
        ret = (feature, label, feature.shape[0])

        return ret

    # Generate label
    def gen_label(self, label_num):

        # Concatenate placeholder
        label_bank = []

        # Loop though label and create numpy array
        for l, n in label_num:

            temp = np.zeros((n,), dtype=np.int32)

            if l in self._label_map:
                temp += self._label_map[l]
            else:
                temp += self._label_map['unknown']

            label_bank.append(temp)

        # Concatenate
        label = np.concatenate(label_bank, axis=0)
        del label_bank

        return label


if __name__ == '__main__':

    print('Running dataset.py as main file...')
