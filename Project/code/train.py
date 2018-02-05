# train.py
# created by hu61
# This file implements the training procedure.
# Including running test and generate the confusion matrix.


import numpy as np
from os import listdir
from scipy.io import loadmat
from scipy.misc import (imsave, imresize)
import tensorflow as tf

from config import (train_config, model_config)
from dataset import DataSet
from feature import feature_generator
from model import RNN


def train(model, trainset, valset, learning_rate=0.0005, num_steps=5000):

    # Start training
    print('\n--------------------------------------------------')
    print('Start training...')
    print('LSTM cell type: {}'.format(model_config.lstm_cell_type))
    print('Hidden units number: {}'.format(model_config.hidden_units_num))
    print('Batch size: {}'.format(train_config.batch_size))
    print('Learning rate: {}'.format(train_config.learning_rate))

    # Prepare saver
    saver = tf.train.Saver()

    # Training step
    for step in range(1, num_steps + 1):

        data, label = trainset.next_batch(train_config.batch_size)

        model.session.run(model.update_op_tensor, feed_dict={
            model.x_placeholder: data,
            model.y_placeholder: label,
            model.batch_size_placeholder: data.shape[0],
            model.learning_rate_placeholder: learning_rate,
        })

        # Validation step
        if step % 100 == 0:

            # Accuracy
            print(model.session.run(model.accuracy, feed_dict={
                model.x_placeholder: valset._data,
                model.y_placeholder: valset._label,
                model.batch_size_placeholder: valset._data.shape[0],
            }))

            # Save
            model.save_model(step)

    # Finish training
    print('Finished training!')
    print('--------------------------------------------------\n')

    return


def test(model, testset):

        # Start testing
    print('\n--------------------------------------------------')
    print('Start testing...')

    # Get y predict and truth
    y_predict = np.argmax(
        model.session.run(model.pred, feed_dict={
            model.x_placeholder: testset._data,
            model.y_placeholder: testset._label,
            model.batch_size_placeholder: testset._data.shape[0],
        }), axis=1)
    y_truth = testset._label

    # Confusion Matrix
    conf_mat = np.zeros((testset._class_num, testset._class_num))

    for y_target in range(testset._class_num):

        total = np.sum(y_truth == y_target)

        for y_result in range(testset._class_num):

            conf_num = np.sum(np.logical_and(y_truth == y_target,
                                             y_predict == y_result))
            conf_mat[y_target, y_result] = conf_num / total

    # Save as image
    imsave('confusion_matrix.png', imresize(
        conf_mat, size=3000, interp='nearest'))

    # logging
    diag_accuracy = np.diag(conf_mat)
    print('Overall accuracy: {}%'.format(
        round(np.mean(diag_accuracy) * 100, 3)))
    for c, i in testset._label_map.items():
        print('>>> class {}: {}%'.format(
            c, round(diag_accuracy[i] * 100, 3)))

    # Finish testing
    print('Finished testing!')
    print('--------------------------------------------------\n')

    return


# Main function
def main():

    # Create Datasets
    train_set = DataSet(key='train')
    val_set = DataSet(key='val')
    test_set = DataSet(key='test')

    # Create RNN model on specific device
    with tf.device(train_config.cuda_device):
        model = RNN(n_dims=train_set._data.shape[2],
                    n_frames=train_set._data.shape[1],
                    n_classes=train_set._class_num,)

    train(model, train_set, val_set,
          learning_rate=train_config.learning_rate,
          num_steps=train_config.num_steps)

    test(model, test_set)

    return

if __name__ == '__main__':
    print('Running train.py as main...')
    main()
