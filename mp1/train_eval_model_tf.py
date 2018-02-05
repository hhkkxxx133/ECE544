"""
Train model and eval model helpers for tensorflow implementation.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression_tf import LinearRegressionTf
# from models.linear_model_tf import LinearModelTf


def train_model(data, model, learning_rate=100, batch_size=16,
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
            data['label'] = np.array(label_data)

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
    model.session.run(model.update_op_tensor, feed_dict = {model.x_placeholder:image_batch, model.y_placeholder:label_batch, model.learning_rate_placeholder:learning_rate})


def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    loss = model.session.run(model.loss_tensor, feed_dict={model.x_placeholder:data['image'], model.y_placeholder:data['label']})
    y_pred = model.session.run(model.predict_tensor, feed_dict={model.x_placeholder:data['image']})
    acc = list(np.multiply(y_pred.flatten(), data['label']) > 0).count(True) / data['label'].shape[0]
    # print("loss: "+str(loss))
    # print("acc: "+str(acc))
    return loss, acc##########
