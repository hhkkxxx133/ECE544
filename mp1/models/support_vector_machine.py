"""
Implements support vector machine.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        N = f.shape[0]
        gt = -np.dot( np.multiply(y,f)<1, np.multiply(np.insert(self.x,self.x.shape[1],1,axis=1), np.expand_dims(y,axis=1)) )
        return gt

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average hinge loss.
        """
        hinge_loss = np.expand_dims(1-np.multiply(y,f),axis=1)
        return np.mean( np.amax( np.insert(hinge_loss,hinge_loss.shape[1], 0, axis=1), axis=1 ) )

    def predict(self, f):
        """
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        return (f>=0).astype(float) - (f<0).astype(float)
