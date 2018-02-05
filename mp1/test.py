"""Unit tests examples for mp1.

Example Usage:
python tests.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from models.linear_regression import LinearRegression


class ModelCheckTest(unittest.TestCase):
    def test_input_output(self):
        model = LinearRegression(10)

        x = np.zeros([4, 10])
        y = np.zeros([4, ])

        # Check forward shape.
        f = model.forward(x)
        self.assertEqual(f.shape, (4,))

        # Check backward shape.
        gradient = model.backward(f, y)
        self.assertEqual(gradient.shape, (11,))

        # Check loss shape.
        loss = model.loss(f, y)
        self.assertEqual(loss.shape, ())


if __name__ == '__main__':
    unittest.main()
