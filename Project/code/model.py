# model.py
# created by hu61
# Implements the RNN model with cell type as BasicLSTMCell
# The model is adaptable to different input size, different
# hidden unit number and different class weights.


import tensorflow as tf
from time import (gmtime, strftime)
import numpy as np

from config import model_config


# RNN Model
class RNN(object):

    def __init__(self, n_dims=28, n_frames=28, n_classes=10,):
        self.batch_size_placeholder = tf.placeholder(tf.int32, [])
        self.n_dims = n_dims
        self.n_frames = n_frames
        self.n_hidden_units = model_config.hidden_units_num
        self.n_classes = n_classes
        self.lstm_cell_type = model_config.lstm_cell_type

        self.weights = {
            # shape (n_dims, 128)
            'in': tf.Variable(tf.random_normal([self.n_dims, self.n_hidden_units])),
            # shape (128, n_classes)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        self.biases = {
            # shape (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # shape (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }

        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32,
                                            [None, self.n_frames, self.n_dims])
        self.y_placeholder = tf.placeholder(tf.int32, [None, ])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        self.pred = self.predict()
        self.accuracy = self.accuracy()
        self.loss_tensor = self.loss()
        self.update_op_tensor = self.update_op()
        self.saver = tf.train.Saver(max_to_keep=model_config.save_model_num)
        self.save_path = model_config.save_path
        self.session.run(tf.global_variables_initializer())

    # Forward Pass
    def predict(self):
        # X ==> (n_batches * n_frames, n_dims)
        X = tf.reshape(self.x_placeholder, [-1, self.n_dims])

        # X_in = W*X + b
        X_in = tf.matmul(X, self.weights['in']) + self.biases['in']
        # X_in ==> (n_batches, n_frames, n_hidden_units)
        X_in = tf.reshape(X_in, [-1, self.n_frames, self.n_hidden_units])

        # use basic LSTM Cell.
        lstm_cell = getattr(tf.contrib.rnn, self.lstm_cell_type)(
            self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)

        init_state = lstm_cell.zero_state(
            self.batch_size_placeholder, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(
            lstm_cell, X_in, initial_state=init_state, time_major=False)

        results = tf.matmul(final_state[1], self.weights[
                            'out']) + self.biases['out']

        return results

    # Cross Entropy Loss Function
    def loss(self):
        weights = tf.where(np.equal(self.y_placeholder, 11),
                           tf.fill([self.batch_size_placeholder, ], 0.167),
                           tf.fill([self.batch_size_placeholder, ], 1.0))
        raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pred, labels=self.y_placeholder)
        weighted_loss = tf.multiply(raw, weights)
        return tf.reduce_mean(weighted_loss)

        # Optimizer Update
    def update_op(self):
        return tf.train.AdamOptimizer(self.learning_rate_placeholder).minimize(self.loss_tensor)

    # Compute accuracy
    def accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.pred, 1, output_type=tf.int32),
                                self.y_placeholder)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    # Helper function to save the models
    def save_model(self, stamp):

        # Obtain Current Time in GMT
        curtime = strftime("%m-%d-%H-%M-", gmtime())
        save_path = self.saver.save(
            self.session, self.save_path + curtime + str(stamp) + '.ckpt')
        print('Model saved in file: {}'.format(save_path))

        return
