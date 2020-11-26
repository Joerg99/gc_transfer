# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 26/03/2020
"""
Main program to train and test the chargrid model

Requirements
----------
- One-hot Chargrid arrays must be located in the folder dir_np_chargrid_1h = "./data/np_chargrids_1h/"
- One-hot Segmentation arrays must be located in the folder dir_np_gt_1h = "./data/np_gt_1h/"
- Bounding Box anchor masks must be located in the folder dir_np_bbox_anchor_mask = "./data/np_bbox_anchor_mask/"
- Bounding Box anchor coordinates must be located in the folder dir_np_bbox_anchor_coord = "./data/np_bbox_anchor_coord/"

Hyperparameters
----------
- (width, height, input_channels) : input shape of one-hot chargrids
- base_channels : number of base channels for the neural network
- (learning_rate, momentum) : parameters of the optimizer
- weight_decay : coefficient used by the l2-regularizer
- spatial_dropout : dropout rate
- nb_classes : number of classes
- proba_classes : probability of each class to appear (classes in this order: other, total, address, company, date)
- constant_weight : constant used to balance class weights
- nb_anchors : number of anchors
- (epochs, batch_size) : parameters for the training process
- prop_test : proportion of the dataset used to validate the model
- seed : seed used to generate randomness (shuffling, ...)
- (pad_left_range, pad_top_range, pad_right_range, pad_bot_range) : padding range used for data augmentation

Return
----------
Several files are generated in the "./output/" folder :
- filename_backup = "./output/model.ckpt" : the model weights for post-training use
- "global_loss.pdf" : plot of the curve containing the global loss functions
- "output_1_loss.pdf" : plot of the curve containing the segmentation loss functions
- "output_2_loss.pdf" : plot of the curve containing the anchor mask loss functions
- "output_3_loss.pdf" : plot of the curve containing the anchor coordinate loss functions
- "train_time.pdf" : plot the time spent to train each epoch
- "test_time.pdf"  : plot the time spent to validate each epoch
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import os
import time
from skimage.transform import resize, rescale
from tensorflow.python.keras.utils.vis_utils import plot_model

## Hyperparameters
dir_np_chargrid_1h = "./data/np_chargrids_1h/"
dir_np_gt_1h = "./data/np_gt_1h/"
dir_np_bbox_anchor_mask = "./data/np_bbox_anchor_mask/"
dir_np_bbox_anchor_coord = "./data/np_bbox_anchor_coord/"
width = 192
height = 192
input_channels = 61 # num of characters
base_channels = 64 # depth of convolutionas layers
learning_rate = 0.05
momentum = 0.9
weight_decay = 0.1 # used for kernel_regularizer. causes that total loss isnt a sum of sub-losses
spatial_dropout = 0.1
nb_classes = 5
#static class weighting
proba_classes = np.array([0.89096397, 0.01125766, 0.0504345, 0.03237164, 0.01497223]) #other, total, address, company, date
constant_weight = 1.04
# num default box per anchor ?
nb_anchors = 9 # one per foreground class
epochs = 10
batch_size_train=8
train_test_split = 0.2 # train/test split
seed = 123456
weights_export = "./output/model.ckpt"
pad_left_range = 0.2
pad_top_range = 0.2
pad_right_range = 0.2
pad_bot_range = 0.2

np.random.seed(seed=seed)

def print_hardware_configuration():
    print(device_lib.list_local_devices())
    print("Tensorflow version: ", tf.__version__)
    print("Test GPU: ", tf.test.gpu_device_name())

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        ## Block z
        self.z1 = tf.keras.layers.Conv2D(input_shape=(None, height, width, input_channels), filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z1_bn = tf.keras.layers.BatchNormalization()

        self.z2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z2_bn = tf.keras.layers.BatchNormalization()

        self.z3 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z3_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z3_bn = tf.keras.layers.BatchNormalization()

        self.z4 = tf.keras.layers.Dropout(rate=spatial_dropout)

        ## Block a
        self.a1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a1_bn = tf.keras.layers.BatchNormalization()

        self.a2 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a2_bn = tf.keras.layers.BatchNormalization()

        self.a3 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a3_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a3_bn = tf.keras.layers.BatchNormalization()

        self.a4 = tf.keras.layers.Dropout(rate=spatial_dropout)


        ## Block a_bis
        self.a_bis_filters = [4*base_channels, 8*base_channels, 8*base_channels]
        self.a_bis_stride = [2, 2, 1]
        self.a_bis_dilatation = [2, 4, 8]

        self.a_bis1 = []
        self.a_bis1_lrelu = []
        self.a_bis1_bn = []

        self.a_bis2 = []
        self.a_bis2_lrelu = []
        self.a_bis2_bn = []

        self.a_bis3 = []
        self.a_bis3_lrelu = []
        self.a_bis3_bn = []

        self.a_bis4 = []

        for i in range(0, len(self.a_bis_filters)):
            self.a_bis1.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=self.a_bis_stride[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis1_bn.append(tf.keras.layers.BatchNormalization())

            self.a_bis2.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=1, dilation_rate=self.a_bis_dilatation[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis2_bn.append(tf.keras.layers.BatchNormalization())

            self.a_bis3.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=1, dilation_rate=self.a_bis_dilatation[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis3_bn.append(tf.keras.layers.BatchNormalization())

            self.a_bis4.append(tf.keras.layers.Dropout(rate=spatial_dropout))


        ## Block b_ss (semantic segmentation)
        self.b_ss_filters = [4*base_channels, 2*base_channels]

        self.b_ss1 = []
        self.b_ss1_lrelu = []
        self.b_ss1_bn = []

        self.b_ss2 = []
        self.b_ss2_lrelu = []
        self.b_ss2_bn = []

        self.b_ss3 = []
        self.b_ss3_lrelu = []
        self.b_ss3_bn = []

        self.b_ss4 = []
        self.b_ss4_lrelu = []
        self.b_ss4_bn = []

        self.b_ss5 = []

        for i in range(0, len(self.b_ss_filters)):
            self.b_ss1.append(tf.keras.layers.Conv2D(filters=2*self.b_ss_filters[i], kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss1_bn.append(tf.keras.layers.BatchNormalization())

            self.b_ss2.append(tf.keras.layers.Conv2DTranspose(filters=self.b_ss_filters[i], kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss2_bn.append(tf.keras.layers.BatchNormalization())

            self.b_ss3.append(tf.keras.layers.Conv2D(filters=self.b_ss_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss3_bn.append(tf.keras.layers.BatchNormalization())

            self.b_ss4.append(tf.keras.layers.Conv2D(filters=self.b_ss_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss4_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss4_bn.append(tf.keras.layers.BatchNormalization())

            self.b_ss5.append(tf.keras.layers.Dropout(rate=spatial_dropout))


        ## Block b_bbr (bounding box regression)
        self.b_bbr_filters = [4*base_channels, 2*base_channels]

        self.b_bbr1 = []
        self.b_bbr1_lrelu = []
        self.b_bbr1_bn = []

        self.b_bbr2 = []
        self.b_bbr2_lrelu = []
        self.b_bbr2_bn = []

        self.b_bbr3 = []
        self.b_bbr3_lrelu = []
        self.b_bbr3_bn = []

        self.b_bbr4 = []
        self.b_bbr4_lrelu = []
        self.b_bbr4_bn = []

        self.b_bbr5 = []

        for i in range(0, len(self.b_bbr_filters)):
            self.b_bbr1.append(tf.keras.layers.Conv2D(filters=2*self.b_bbr_filters[i], kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr1_bn.append(tf.keras.layers.BatchNormalization())

            self.b_bbr2.append(tf.keras.layers.Conv2DTranspose(filters=self.b_bbr_filters[i], kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr2_bn.append(tf.keras.layers.BatchNormalization())

            self.b_bbr3.append(tf.keras.layers.Conv2D(filters=self.b_bbr_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr3_bn.append(tf.keras.layers.BatchNormalization())

            self.b_bbr4.append(tf.keras.layers.Conv2D(filters=self.b_bbr_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr4_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr4_bn.append(tf.keras.layers.BatchNormalization())

            self.b_bbr5.append(tf.keras.layers.Dropout(rate=spatial_dropout))


        ## Block c_ss
        self.c_ss1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_ss1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_ss1_bn = tf.keras.layers.BatchNormalization()

        self.c_ss2 = tf.keras.layers.Conv2DTranspose(filters=base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_ss2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_ss2_bn = tf.keras.layers.BatchNormalization()


        ## Block c_bbr
        self.c_bbr1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_bbr1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_bbr1_bn = tf.keras.layers.BatchNormalization()

        self.c_bbr2 = tf.keras.layers.Conv2DTranspose(filters=base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_bbr2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_bbr2_bn = tf.keras.layers.BatchNormalization()


        ## Block d
        self.d1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.d1_bn = tf.keras.layers.BatchNormalization()

        self.d2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.d2_bn = tf.keras.layers.BatchNormalization()

        self.d3 = tf.keras.layers.Conv2D(filters=nb_classes, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d3_softmax = tf.keras.layers.Softmax(name="d3_softmax_out")


        ## Block e
        self.e1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.e1_bn = tf.keras.layers.BatchNormalization()

        self.e2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.e2_bn = tf.keras.layers.BatchNormalization()

        self.e3 = tf.keras.layers.Conv2D(filters=2*nb_anchors, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e3_softmax = tf.keras.layers.Softmax(name="e3_softmax_out")


        ## Block f
        self.f1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.f1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.f1_bn = tf.keras.layers.BatchNormalization()

        self.f2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.f2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.f2_bn = tf.keras.layers.BatchNormalization()

        self.f3 = tf.keras.layers.Conv2D(filters=4*nb_anchors, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))

        self.rpn_cls_conv = tf.keras.layers.Conv2D(filters=8*base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.rpn_cls_relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.rpn_cls_bn = tf.keras.layers.BatchNormalization()

        '''
        filters = 2* anchors if 1hot encoding per anchbor box use activation = softmax
        filters = 1* anchors if no 1hot encoding use activation = sigmoid
        '''
        self.rpn_cls_out = tf.keras.layers.Conv2D(filters=1*nb_anchors, activation="sigmoid", kernel_size=1, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay), name="rpn_cls_out")



    def build_graph(self, in_shape):
        x = tf.keras.Input(shape=(in_shape))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    # call is constructing the network in keras-sequential style
    # return value are the outputs of the model
    def call(self, input):
        ## Encoder
        x = self.z1(input)
        x = self.z1_lrelu(x)
        x = self.z1_bn(x)
        x = self.z2(x)
        x = self.z2_lrelu(x)
        x = self.z2_bn(x)
        x = self.z3(x)
        x = self.z3_lrelu(x)
        x = self.z3_bn(x)
        out_z = self.z4(x)

        x = self.a1(out_z)
        x = self.a1_lrelu(x)
        x = self.a1_bn(x)
        x = self.a2(x)
        x = self.a2_lrelu(x)
        x = self.a2_bn(x)
        x = self.a3(x)
        x = self.a3_lrelu(x)
        x = self.a3_bn(x)
        out_a = self.a4(x)

        out_a_bis = []
        x = out_a
        for i in range(0, len(self.a_bis_filters)):
            x = self.a_bis1[i](x)
            x = self.a_bis1_lrelu[i](x)
            x = self.a_bis1_bn[i](x)
            x = self.a_bis2[i](x)
            x = self.a_bis2_lrelu[i](x)
            x = self.a_bis2_bn[i](x)
            x = self.a_bis3[i](x)
            x = self.a_bis3_lrelu[i](x)
            x = self.a_bis3_bn[i](x)
            x = self.a_bis4[i](x)
            out_a_bis.append(x)

        ## Decoder Semantic Segmentation
        """
        concat_tab = [out_a_bis[1], out_a_bis[0]]
        for i in range(0, len(self.b_ss_filters)):
            x = tf.concat([x, concat_tab[i]], 3)
            x = self.b_ss1[i](x)
            x = self.b_ss1_lrelu[i](x)
            x = self.b_ss1_bn[i](x)
            x = self.b_ss2[i](x)
            x = self.b_ss2_lrelu[i](x)
            x = self.b_ss2_bn[i](x)
            x = self.b_ss3[i](x)
            x = self.b_ss3_lrelu[i](x)
            x = self.b_ss3_bn[i](x)
            x = self.b_ss4[i](x)
            x = self.b_ss4_lrelu[i](x)
            x = self.b_ss4_bn[i](x)
            x = self.b_ss5[i](x)

        x = tf.concat([x, out_a], 3)
        x = self.c_ss1(x)
        x = self.c_ss1_lrelu(x)
        x = self.c_ss1_bn(x)
        x = self.c_ss2(x)
        x = self.c_ss2_lrelu(x)
        x = self.c_ss2_bn(x)

        x = self.d1(x)
        x = self.d1_lrelu(x)
        x = self.d1_bn(x)
        x = self.d2(x)
        x = self.d2_lrelu(x)
        x = self.d2_bn(x)
        x = self.d3(x)
        out_d = self.d3_softmax(x)
        """

        ## Decoder Bounding Box Regression
        """
        concat_tab = [out_a_bis[1], out_a_bis[0]]
        x = out_a_bis[-1]
        for i in range(0, len(self.b_bbr_filters)):
            x = tf.concat([x, concat_tab[i]], 3)
            x = self.b_bbr1[i](x)
            x = self.b_bbr1_lrelu[i](x)
            x = self.b_bbr1_bn[i](x)
            x = self.b_bbr2[i](x)
            x = self.b_bbr2_lrelu[i](x)
            x = self.b_bbr2_bn[i](x)
            x = self.b_bbr3[i](x)
            x = self.b_bbr3_lrelu[i](x)
            x = self.b_bbr3_bn[i](x)
            x = self.b_bbr4[i](x)
            x = self.b_bbr4_lrelu[i](x)
            x = self.b_bbr4_bn[i](x)
            x = self.b_bbr5[i](x)

        x = tf.concat([x, out_a], 3)
        x = self.c_bbr1(x)
        x = self.c_bbr1_lrelu(x)
        x = self.c_bbr1_bn(x)
        x = self.c_bbr2(x)
        x = self.c_bbr2_lrelu(x)
        out_c_bbr = self.c_bbr2_bn(x)

        x = self.e1(out_c_bbr)
        x = self.e1_lrelu(x)
        x = self.e1_bn(x)
        x = self.e2(x)
        x = self.e2_lrelu(x)
        x = self.e2_bn(x)
        x = self.e3(x)
        out_e = self.e3_softmax(x)

        x = self.f1(out_c_bbr)
        x = self.f1_lrelu(x)
        x = self.f1_bn(x)
        x = self.f2(x)
        x = self.f2_lrelu(x)
        x = self.f2_bn(x)
        out_f = self.f3(x)
        """
        # return out_d, out_e, out_f

        # Region Proposal Net

        x = self.rpn_cls_conv(x)
        x = self.rpn_cls_relu(x)
        x = self.rpn_cls_bn(x)
        out_rpn_cls = self.rpn_cls_out(x)

        return out_rpn_cls


def augment_data(data, tab_rand, order, shape, coord=False):
    '''
    no augmentation in the typical computer vision meaning
    :return:
    '''
    data_temp = resize(np.pad(data, ((tab_rand[1], tab_rand[3]), (tab_rand[0], tab_rand[2]), (0, 0)), 'constant'), shape, order=order, anti_aliasing=True)
    
    if coord:
        for i in range(0, nb_anchors):
            mask = (data_temp > 1e-6)[:, :, 4*i]
            data_temp[mask, 4*i] *= shape[1]
            data_temp[mask, 4*i] += tab_rand[0]
            data_temp[mask, 4*i] /= (tab_rand[0]+shape[1]+tab_rand[2])
            
            data_temp[mask, 4*i+2] *= shape[1]
            data_temp[mask, 4*i+2] += tab_rand[0]
            data_temp[mask, 4*i+2] /= (tab_rand[0]+shape[1]+tab_rand[2])
            
            data_temp[mask, 4*i+1] *= shape[0]
            data_temp[mask, 4*i+1] += tab_rand[1]
            data_temp[mask, 4*i+1] /= (tab_rand[1]+shape[0]+tab_rand[3])
            
            data_temp[mask, 4*i+3] *= shape[0]
            data_temp[mask, 4*i+3] += tab_rand[1]
            data_temp[mask, 4*i+3] /= (tab_rand[1]+shape[0]+tab_rand[3])

    return data_temp


def get_class_weights():
    '''
    uses global variables constant_weight and proba_classes
    calculates static class weighting as classes are imbalanced
    :return:
    sample_weight: the weights for how a sample is weighted during metric and
        loss calculation. Could be None.
    '''
    sample_weight_seg = np.ones((height, width, nb_classes))*1.0/np.log(constant_weight+proba_classes)

    proba_classes_boxmask = np.repeat(proba_classes[1:], 2)
    proba_classes_boxmask[np.arange(1, 2*nb_anchors, 2)] = 1-proba_classes[1:]
    sample_weight_boxmask = np.ones((height, width, 2*nb_anchors))*1.0/np.log(constant_weight+proba_classes_boxmask)
    
    return sample_weight_seg, sample_weight_boxmask



def get_train_test_sets(list_filenames):
    np.random.shuffle(list_filenames)
    
    trainset = list_filenames[int(len(list_filenames) * train_test_split):]
    testset = list_filenames[:int(len(list_filenames) * train_test_split)]
    
    return trainset, testset

def compare_input_augmented_input(index_to_test, trainset, batch_chargrid, batch_seg, batch_mask, batch_coord):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.apply_along_axis(np.argmax, axis=2, arr=np.load(os.path.join(dir_np_chargrid_1h, trainset[index_to_test]))))
    ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=batch_chargrid[index_to_test]))
    plt.show()
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.apply_along_axis(np.argmax, axis=2, arr=np.load(os.path.join(dir_np_gt_1h, trainset[index_to_test]))))
    ax2.imshow(np.apply_along_axis(np.argmax, axis=2, arr=batch_seg[index_to_test]))
    plt.show()
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.load(os.path.join(dir_np_bbox_anchor_mask, trainset[index_to_test]))[:, :, 0])
    ax2.imshow(batch_mask[index_to_test][:, :, 0])
    plt.show()
    plt.clf()

    print(batch_coord[index_to_test][(batch_coord[index_to_test] > 1e-6)[:, :, 0], 0]*width)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.load(os.path.join(dir_np_bbox_anchor_coord, trainset[index_to_test]))[:, :, 0])
    ax2.imshow(batch_coord[index_to_test][:, :, 0])
    plt.show()
    plt.clf()

def plot_loss(history_train, history_test, title, filename):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(history_train, label="train")
    plt.plot(history_test, label="val")
    plt.plot(np.argmin(history_test), np.min(history_test), marker="o", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="pdf")
    plt.close()

def plot_time(history_time, title, filename):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(history_time, label="time")
    plt.xlabel("Epochs")
    plt.ylabel("Time")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, format="pdf")
    plt.close()


def initialize_network(sample_weight_seg, sample_weight_boxmask):
    net = Network()

    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = tf.keras.backend.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            # calc
            loss = y_true * tf.keras.backend.log(y_pred) * weights
            loss = -tf.keras.backend.sum(loss, -1)
            return loss

        return loss
    custom_loss = weighted_categorical_crossentropy(np.array([1,10]))

    net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False),
                loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryCrossentropy()] ) #loss=tf.keras.losses.BinaryCrossentropy()


    # .build only needed for subclassing model
    net.build(input_shape=(None, height, width, input_channels))

    return net

def extract_batch(dataset, batch_size, pad_left_range, pad_top_range, pad_right_range, pad_bot_range):
    '''
    :return:  X = batch_chargrid
              Y = batch_seg, batch_mask, batch_coord

              dimensions: batch, 256 , 128 (reduced img size), depth (chargrid=61, segment= 5, mask = 8, coord=16)
    '''
    if batch_size > len(dataset):
        raise NameError('batch_size > len(dataset)')
    np.random.shuffle(dataset)

    tab_rand = np.random.rand(batch_size, 4) * [pad_left_range * width, pad_top_range * height, pad_right_range * width,
                                                pad_bot_range * height]
    tab_rand = tab_rand.astype(int)

    chargrid_input = []
    seg_gt = []
    anchor_mask_gt = []
    anchor_coord_gt = []

    for i in range(0, batch_size):
        data = np.load(os.path.join(dir_np_chargrid_1h, dataset[i]))
        chargrid_input.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, input_channels)))

        data = np.load(os.path.join(dir_np_gt_1h, dataset[i]))
        seg_gt.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, nb_classes)))

        data = np.load(os.path.join(dir_np_bbox_anchor_mask, dataset[i]))
        anchor_mask_gt.append(augment_data(data, tab_rand[i], order=1, shape=(height, width, 2 * nb_anchors)))

        data = np.load(os.path.join(dir_np_bbox_anchor_coord, dataset[i]))
        anchor_coord_gt.append(
            augment_data(data, tab_rand[i], order=0, shape=(height, width, 4 * nb_anchors), coord=True))

    return np.array(chargrid_input), np.array(seg_gt), np.array(anchor_mask_gt), np.array(anchor_coord_gt)


def train(net):
    history_loss = []
    history_acc = []
    history_val_loss = []
    history_val_acc = []

    def lr_scheduler(epoch):
        base_lr = 0.05
        if epoch == 0:
            return base_lr
        elif epoch < 5:
            return base_lr / (epoch * 5)
        else:
            return 0.00005

    callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    train_img, train_labels_cls = get_data(num_samples=24)    #save_folder = f"./output/ss_only/005_dice_1_1_2_2_1weights_allsamples_lrsched_small_model"


    for epoch in range(epochs):
        print(f"Start Epoch {epoch}")
        # Training

        history = net.fit(x=train_img, y=train_labels_cls, shuffle=True, batch_size=batch_size_train, verbose=1, callbacks=[callback], validation_split=0.2)

        history_loss.append(history.history["loss"])
        history_acc.append(history.history["binary_crossentropy"])
        history_val_loss.append(history.history["val_loss"])
        history_val_acc.append(history.history["val_binary_crossentropy"])

        net.save_weights(f"./output/bbox_cls/model_ep_{epoch}.ckpt")
        # if (epoch % 5) == 0:
        #results = net.predict(batch_chargrid)
        # print(results[0, 45:54, 16:18, :]) #for dummy data
        # print(results[0, 80:90, 40:41, :])  # for X51008123604.npy three different labels occur in this line
        #
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(results[0, :, :, :].argmax(axis=2))
        # ax2.imshow(results[1, :, :, :].argmax(axis=2))
        # ax3.imshow(results[2, :, :, :].argmax(axis=2))
        # plt.savefig(f"{save_folder}/predictions_ep_{epoch}.png")
        # plt.close()

    return history_loss, history_acc, history_val_loss, history_val_acc

def plot_subclass_model():
    # https://stackoverflow.com/questions/61427583/how-do-i-plot-a-keras-tensorflow-subclassing-api-model
    Network().build_graph(in_shape=(height, width, input_channels)).summary()
    plot_model(Network().build_graph(in_shape=(height, width, input_channels)),show_shapes=True, show_layer_names=True, to_file="archi.png")
    print("archi.png created")

def get_data(num_samples):
    import random

    one_hot = r"C:\workspace\Chargrid\data\\bbox_labels_cls_min_1_iou"
    not_one_hot = r"C:\workspace\Chargrid\data\\bbox_labels_cls_min_1_iou_not_1_hot"
    dir_to_open = not_one_hot
    list_filenames = [f for f in os.listdir(dir_to_open) if
                      os.path.isfile(os.path.join(dir_to_open, f))]

    train_samples = []
    for example in list_filenames:
        im = np.load(f"C:\workspace\Chargrid\data\\np_chargrid_unscaled_top_1h\\{example}")
        label = np.load(f"{dir_to_open}\\{example}")
        train_samples.append((im, label))

    random.shuffle(train_samples)
    train_samples = list(zip(*train_samples))
    train_input = train_samples[0]
    train_labels = train_samples[1]
    return tf.stack(train_input[:num_samples]), tf.stack(train_labels[:num_samples])

if __name__ == "__main__":

    train_flag= False
    if train_flag:
        net = initialize_network(None, None)
        #net.load_weights("./output/model.ckpt")
        plot_subclass_model()
        history_loss, history_acc, history_val_loss, history_val_acc = train(net)
        #net.save_weights(f"{weights_export}") # save final weights
        ## Plot loss
        plot_loss(history_loss, history_val_loss, "Global Loss", "./output/bbox_cls/global_loss.pdf")
        plot_loss(history_acc, history_val_acc, "Acc", "./output/bbox_cls/acc_cls.pdf")
    else:
        import heapq
        img_input, label = get_data(num_samples=4)
        net = initialize_network(None, None)
        net.load_weights("./output/bbox_cls/model_ep_7.ckpt")
        predictions = net.predict(x=img_input)

        print()
        #weights = tf.keras.backend.variable([1,1,1,1,1])
        # x, y = get_data(2)
        y_true = label #np.array([1,1,0,0,1], dtype=np.double)
        y_pred = predictions # np.array([0,0,1,1,0], dtype=np.double)
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred)  #* np.array([0,0,0,0,1], dtype=np.double)
        loss = -tf.keras.backend.sum(loss, -1)
        loss
        print()

        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_loss = bce(y_true,y_pred)
        bce_loss.shape

    #
    #     print(3702)
    #     """
    #     true layer should be 1 for proposed bbox
    #     print index where labels are 1
    #     print
    #     """
    #     true_layer = [0,2,4,6,8,10,12,14,16]
    #     for i in true_layer:
    #         print(heapq.nlargest(3, range(len(np.reshape(predictions[0, :, :, i], 12 * 12))), np.reshape(predictions[0, :, :, i], 12 * 12).take))
    #         print(np.where(np.reshape(label[0, :, :, i], 12 * 12) == 1))
    #     predictions[0,:,:,0]