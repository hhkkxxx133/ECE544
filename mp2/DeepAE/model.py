################################################################################
# Author: Safa Messaoud                                                        #
# E-Mail: messaou2@illinois.edu                                                #
# Instituation: University of Illinois at Urbana-Champaign                     #
# Course: ECE 544_na Fall 2017                                                 #
# Date: July 2017                                                              #
#                                                                              #
# Description: the denoising convolutional autoencoder model                   #
#                                                                              #
#                                                                              #
################################################################################

import tensorflow as tf
import numpy as np
import utils


class DAE(object):
    """
    Denoising Convolutional Autoencoder
    """

    def __init__(self, config):
        """
        Basic setup.
        Args:
            config: Object containing configuration parameters.
        """

        # Model configuration.
        self.config = config    	

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.original_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.noisy_images = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.reconstructed_images = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # Global step Tensor.
        self.global_step = None

        # A boolean indicating whether the current mode is 'training'.
        self.phase_train = True



    
  
    def build_inputs(self):
        """ Input Placeholders.
        define place holders for feeding (1) noise-free images, (2) noisy images and (3) a boolean variable 
        indicating whether you are in the training or testing phase
        Outputs:
            self.original_images
            self.noisy_images
            self.phase_train
        """
        
        
        self.original_images = tf.placeholder(tf.float32,[None,28*28],'images') #images
        self.noisy_images = tf.placeholder(tf.float32,[None,28*28],'noisy_images') #noisy_images
        self.phase_train = tf.placeholder(tf.bool,[],'phase_train') #phase_train



    def build_model(self):
        """Builds the model.
        # implements the denoising auto-encoder. Feel free to experiment with different architectures.
        Explore the effect of 1) deep networks (i.e., more layers), 2) interlayer batch normalization and
        3) dropout, 4) pooling layers, 5) convolution layers, 6) upsampling methods (upsampling vs deconvolution), 
        7) different optimization methods (e.g., stochastic gradient descent versus stochastic gradient descent
        with momentum versus RMSprop.  
        Do not forget to scale the final output between 0 and 1. 
        Inputs:
            self.noisy_images
            self.original_images
        Outputs:
            self.total_loss
            self.reconstructed_images 
        """  
        # original: 28x28x1
        ### Encoder
        # def net(self):
        #     with tf.name_scope('reshape'):
        #         x_noisy = tf.reshape(self.noisy_images, [-1,28,28,1])

        #     with tf.name_scope('conv1'):
        #         w_conv1 = tf.Variable(tf.random_normal([ndims+1], tf.float32))
        #         z_conv1 = tf.nn.relu(tf.nn.conv2d()+b1)

        #     with tf.name_scope('pool1'):
        #         z_pool1 = tf.nn.max_pool()

        ### Encoder
        x_noisy = tf.reshape(self.noisy_images, [-1,28,28,1])
        conv1 = tf.layers.conv2d(inputs=x_noisy, filters=32, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv1_norm = utils.batch_norm(conv1, 32, self.phase_train)
        conv1_out = tf.nn.relu(conv1_norm)
        # 20*28x28x32
        pool1 = tf.layers.max_pooling2d(conv1_out, pool_size=(2,2), strides=(2,2), padding='same')
        # 20*14x14x32
        conv2a = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv2a_norm = utils.batch_norm(conv2a, 16, self.phase_train)
        conv2b = tf.layers.conv2d(inputs=conv2a_norm, filters=16, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv2b_norm = utils.batch_norm(conv2b, 16, self.phase_train)
        conv2b_out = tf.nn.relu(conv2b_norm)
        # 20*14x14x16
        pool2 = tf.layers.max_pooling2d(conv2b_out, pool_size=(2,2), strides=(2,2), padding='same')
        # 20*7x7x16
        conv3a = tf.layers.conv2d(inputs=pool2, filters=8, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv3a_norm = utils.batch_norm(conv3a, 8, self.phase_train)
        conv3b = tf.layers.conv2d(inputs=conv3a_norm, filters=8, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv3b_norm = utils.batch_norm(conv3b, 8, self.phase_train)
        conv3b_out = tf.nn.relu(conv3b_norm)
        # 20*7x7x8
        encoded = tf.layers.max_pooling2d(conv3b_out, pool_size=(2,2), strides=(2,2), padding='same')
        # 20*4x4x8

        ### Decoder
        upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 20*7x7x8
        conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv4_norm = utils.batch_norm(conv4, 8, self.phase_train)
        conv4_out = tf.nn.relu(conv4_norm)
        # 20*7x7x8
        upsample2 = tf.image.resize_images(conv4_out, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 20*14x14x8
        conv5 = tf.layers.conv2d(inputs=upsample2, filters=16, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv5_norm = utils.batch_norm(conv5, 16, self.phase_train)
        conv5_out = tf.nn.relu(conv5_norm)
        # 20*14x14x16
        upsample3 = tf.image.resize_images(conv5_out, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # 20*28x28x16
        conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same')#, activation=tf.nn.relu)
        conv6_norm = utils.batch_norm(conv6, 32, self.phase_train)
        conv6_out = tf.nn.relu(conv6_norm)
        # 20*28x28x32

        conv7 = tf.layers.conv2d(inputs=conv6_out, filters=1, kernel_size=(3,3), padding='same')#, activation=None)
        conv7_norm = utils.batch_norm(conv7, 1, self.phase_train)
        conv7_out = tf.nn.relu(conv7_norm)
        # 20*28x28x1

        decoded = tf.nn.sigmoid(conv7)
        x_reconstructed = tf.reshape(decoded, tf.shape(self.original_images))
        self.reconstructed_images = x_reconstructed

        # Compute losses.
        self.total_loss = tf.sqrt(tf.reduce_mean(tf.square(x_reconstructed - self.original_images)))


       

    def setup_global_step(self):
	    """Sets up the global step Tensor."""
	    global_step = tf.Variable(
	    	initial_value=0,
	        name="global_step",
	        trainable=False,
	        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

	    self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_model()
        self.setup_global_step()


