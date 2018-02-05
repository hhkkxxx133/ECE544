"""Variation autoencoder."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers


class VariationalAutoencoder(object):
    """Varational Autoencoder.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a VAE

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Create session
        self.session = tf.Session()
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.z_mean, self.z_log_var = self._encoder(self.x_placeholder)
        self.z = self._sample_z(self.z_mean, self.z_log_var)
        self.outputs_tensor = self._decoder(self.z)

        # Setup loss tensor, predict_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.outputs_tensor, self.x_placeholder,
                                     self.z_mean, self.z_log_var)

        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())

    def _sample_z(self, z_mean, z_log_var):
        """Sample z using reparametrize trick.

        Args:
            z_mean (tf.Tensor): The latent mean,
                tensor of dimension (None, 2)
            z_log_var (tf.Tensor): The latent log variance,
                tensor of dimension (None, 2)
        Returns:
            z (tf.Tensor): Random z sampled of dimension (None, 2)
        """
        std = tf.sqrt(tf.exp(z_log_var))
        sample = tf.random_normal(tf.shape(z_mean),mean=0,stddev=1)
        z = z_mean + std * sample
        return z

    def _encoder(self, x):
        """Encoder block of the network.

        Build a two layer network of fully connected layers, with 100 nodes,
        then 50 nodes. Then two output branches each two 2 nodes representing
        the z_mean and z_log_var.

                             |-> 2 (z_mean)
        Input --> 100 --> 50 -
                             |-> 2 (z_log_var)

        Use activation of tf.nn.softplus.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
        Returns:
            z_mean(tf.Tensor): The latent mean, tensor of dimension (None, 2).
            z_log_var(tf.Tensor): The latent log variance, tensor of dimension
                (None, 2).
        """
        en_fc1 = tf.layers.dense(inputs=x, units=100, activation=tf.nn.softplus)
        en_fc2 = tf.layers.dense(inputs=en_fc1, units=50, activation=tf.nn.softplus)
        z_mean= tf.layers.dense(inputs=en_fc2, units=2)#, activation=tf.nn.softplus)
        z_log_var= tf.layers.dense(inputs=en_fc2, units=2)#, activation=tf.nn.softplus)
        return z_mean, z_log_var

    def _decoder(self, z):
        """From a sampled z, decode back into image.

        Build a three layer network of fully connected layers,
        with 50, 100, 784 nodes. Use activation tf.nn.softplus.
        z (2) --> 50 --> 100 --> 784.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
        Returns:
            f(tf.Tensor): Predicted score, tensor of dimension (None, 784).
        """
        de_fc1 = tf.layers.dense(inputs=z, units=50, activation=tf.nn.softplus)
        de_fc2 = tf.layers.dense(inputs=de_fc1, units=100, activation=tf.nn.softplus)
        f = tf.layers.dense(inputs=de_fc2, units=784)#, activation=tf.nn.softplus)
        return f

    def _latent_loss(self, z_mean, z_log_var):
        """Constructs the latent loss.

        Args:
            z_mean(tf.Tensor): Tensor of dimension (None, 2)
            z_log_var(tf.Tensor): Tensor of dimension (None, 2)

        Returns:
            latent_loss: Tensor of dimension (None,). Sum the loss over
            dimension 1.
        """
        latent_loss = 0.5*tf.reduce_sum(tf.square(z_mean) + tf.exp(z_log_var) - z_log_var - 1, axis=1)
        return latent_loss

    def _reconstruction_loss(self, f, y):
        """Constructs the reconstruction loss.
        Args:
            f(tf.Tensor): Predicted score for each example, dimension (None,
                784).
            y(tf.Tensor): Ground truth for each example, dimension (None, 784)
        Returns:
            Tensor for dimension (None,). Sum the loss over dimension 1.
        """
        return tf.reduce_sum(tf.square(f-y), axis=1)

    def loss(self, f, y, z_mean, z_var):
        """Computes the total loss.

        Computes the sum of latent and reconstruction loss then average over
        dimension 0.

        Returns:
            (1) averged loss of latent_loss and reconstruction loss over
                dimension 0.
        """
        return tf.reduce_mean(self._latent_loss(z_mean, z_var) + self._reconstruction_loss(f,y))

    def update_op(self, loss, learning_rate):
        """Creates the update optimizer.

        Use tf.train.AdamOptimizer to obtain the update op.

        Args:
            loss: Tensor containing the loss function.
            learning_rate: A scalar, learning rate for gradient descent.
        Returns:
            (1) Update opt tensorflow operation.
        """
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def generate_samples(self, z_np):
        """Generates a random sample from the provided z_np.

        Args:
            z_np(numpy.ndarray): Numpy array of dimension (batch_size, 2).

        Returns:
            (1) The sampled images (numpy.ndarray) of dimension (batch_size,
                748).
        """
        # sample = tf.stack( (self.outputs_tensor - tf.reduce_mean(self.outputs_tensor)) / tf.reduce_max(self.outputs_tensor) )
        sample = self.session.run(self.outputs_tensor, feed_dict={self.z:z_np})
        # sample = np.stack( ((sample[i,:] - np.min(sample[i,:])) / np.max(sample[i,:])  for i in range(sample.shape[0])) )
        sample -= np.min(sample)
        sample /= np.max(sample)
        return sample
