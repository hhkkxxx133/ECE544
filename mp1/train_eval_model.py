"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    batch_epoch_num = data['label'].shape[0] // batch_size

    for i in range(num_steps):

        if shuffle == True and i%batch_epoch_num == 0:
            idx = np.array( list(range(data['label'].shape[0])) )
            np.random.shuffle(idx)
            data_new = {}
            label_data = []
            img_data = np.zeros( (data['label'].shape[0], 28*28) )
            for k,val in enumerate(idx):
                label_data.append(data['label'][val])
                img_data[k,:] = data['image'][val,:]
            data['image'] = img_data
            data['label'] = np.asarray(label_data)

        image_batch = data['image'][ (i%batch_epoch_num)*batch_size : (i%batch_epoch_num+1)*batch_size , : ]
        label_batch = data['label'][ (i%batch_epoch_num)*batch_size : (i%batch_epoch_num+1)*batch_size ]
        update_step(image_batch, label_batch, model, learning_rate)

    return model


def update_step(image_batch, label_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).
    Args:
        image_batch(numpy.ndarray): input data of dimension (N, ndims).
        label_batch(numpy.ndarray): label data of dimension (N,).
        model(LinearModel): Initialized linear model.
    """
    # print(image_batch[0][:50])
    f = model.forward(image_batch)
    gt = model.backward(f, label_batch)
    model.w -= learning_rate * gt


def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    y_predict = model.predict(model.forward(data['image']))
    loss = model.loss(model.forward(data['image']), data['label'])
    acc = list(np.multiply(y_predict, data['label']) > 0).count(True) / data['label'].shape[0]
    # print("loss: " + str(loss))
    # print("acc: "+ str(acc))
    return loss, acc
