"""Linear model base class for Tensorflow implementation.
"""

import abc
import numpy as np
import tensorflow as tf

class LinearModelTf(object):
    def __init__(self, ndims, w_init='zeros'):
        """
        """
        self.w = None
        self.ndims = ndims
        if w_init == 'zeros':
            self.w = tf.Variable(tf.zeros(ndims+1, tf.float32))
        elif w_init == 'ones':
            self.w = tf.Variable(tf.ones(ndims+1, tf.float32))
        elif w_init == 'uniform':
            self.w = tf.Variable(tf.random_uniform([ndims+1], tf.float32))
        elif w_init == 'gaussian':
            self.w = tf.Variable(tf.random_normal([ndims+1], tf.float32))

        # You do not have the change the code below in this function.
        # Create Session.
        self.session = tf.Session()

        # Create Placeholders.
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        self.y_placeholder = tf.placeholder(tf.float32, [None])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        outputs = self.forward(self.x_placeholder)

        self.loss_tensor = None
        self.predict_tensor = None
        self.update_op_tensor = None

        # Setup loss_tensor, predict_tensor, update_op_tensor.
        self.loss_tensor = self.loss(outputs, self.y_placeholder)
        self.predict_tensor = self.predict(outputs)
        self.update_op_tensor = self.update_op(
            self.loss_tensor,
            self.learning_rate_placeholder
        )

        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())

    def forward(self, x):
        """Forward operation for linear models.
        Performs the forward opreration, f=w^tx, and return f.

        Args:
            x(tf.placeholder): Tensorflow placeholder of (None, ndims).
        Returns:
            f(tf.Tensor): Tensor for dimension (None, 1)
        """
        bias = tf.ones(tf.stack([tf.shape(x)[0],1]))
        f = tf.matmul( tf.concat( [x, bias], 1 ), tf.expand_dims(self.w,1) )
        return f

    def update_op(self, loss, learning_rate):
        """Backward update operation for linear models.

        Use tf.train.GradientDescentOptimizer to obtain the update op.

        Args:
            loss: Tensor containing the loss function.
            learning_rate: A scalar, learning rate for gradient descent.
        Returns: 
            (1) GradientDescent tensorflow operation.
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        return optimizer

    @abc.abstractmethod
    def loss(self, f, y):
        """Do not need to be implemented here."""
        pass

    @abc.abstractmethod
    def predict(self, f):
        """Do not need to be implemented here."""
        pass
