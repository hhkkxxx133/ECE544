"""
Implements lienar regression.
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model"""

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
        gt = -2 / N * np.dot( np.multiply( (np.ones(N) - np.multiply(y, f)), y ) , np.insert(self.x, self.x.shape[1],1,axis=1) )
        return gt

    def loss(self, f, y):
        """The average loss across batch examples.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
            y(numpy.ndarray): Ground truth label, dimension (N,).
        Returns:
            (float): average square loss.
        """
        # N = f.shape[0]
        # loss = 1 / N * np.sum( np.square(y-f) )
        return np.mean(np.square(y-f))

    def predict(self, f):
        """Converts score into predictions in {-1, 1}.
        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,).
        """
        # y_predict = np.array( [1 if f[idx]>0 else -1 for idx in range(len(f))] )
        return (f>=0).astype(float) - (f<0).astype(float)
