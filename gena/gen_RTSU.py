# -*- coding: utf-8 -*-

""" Wide Residual Network.

Applying a Wide Residual Network to CIFAR-10 Dataset classification task.

References:
    - Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
    - wide Residual Network
Links:
    - [wide Residual Network]
    - [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""

from __future__ import division, print_function, absolute_import
import sys
import argparse
import tflearn
import numpy as np
from PIL import Image
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.activations import shift_RSU
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
import os
import random
import math
try:
    import scipy.ndimage
except Exception:
    print("Scipy not supported!")
    
dropout_probability = 0

def random_shuffle_image(batch):
    size, w,  h, channel = batch.shape
    batch = batch.reshape(size, w*h*channel)
    for i in xrange(size):
        idx = np.arange(w*h*channel)
        np.random.shuffle(idx)
        batch[i, :] = batch[i, idx]

    return batch.reshape(size, w, h, channel)

def random_shuffle(batch):
    size, w,  h, channel = batch.shape
    batch = batch.reshape(size, w*h, channel)
    for i in xrange(size):
        for j in xrange(channel):
            idx = np.arange(w*h)
            np.random.shuffle(idx)
            batch[i, :, j] = batch[i, idx, j]
    
    return batch.reshape(size, w, h, channel)

def roll(batch):
    size, w,  h, channel = batch.shape
    batch = batch.reshape(size, w*h, channel)
    for i in xrange(size):
        for j in xrange(channel):
            roll_size = random.randint(0, w*h)
            batch[i, :, j] = np.roll(batch[i, :, j], roll_size)
    
    return batch.reshape(size, w, h, channel)

def random_flip_leftright(batch):
    for i in range(len(batch)):
        batch[i] = np.fliplr(batch[i])
    return batch

def random_rotation(batch, max_angle):
    for i in range(len(batch)):
        # Random angle
        angle = random.uniform(-max_angle, max_angle)
        batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle,
                                                      reshape=False)
        
    return batch

def median_filter(batch):
    for i in range(len(batch)):
        batch[i] = scipy.ndimage.filters.median_filter(batch[i], 3)
    return batch

def gaussian_filter(batch):
    for i in range(len(batch)):
        batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma=(0.5, 0.5, 0), order=0)
    return batch

def affine_transform(batch):
    cin = 0.5*np.array(batch.shape[1:-1])
    dest_shape = batch.shape[1:-1]
    cout = cin
    theta = random.uniform(-180, 180)
    rot = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
    invRot = rot.T
    invscale = np.diag((1+random.normalvariate(0, 0.3), 1+random.normalvariate(0, 0.3)))
    invTransform = np.dot(invscale, invRot)
    offset = cin - np.dot(invTransform, cout)
    for i in range(len(batch)):
        r = batch[i, :, :, 0]
        g = batch[i, :, :, 1]
        b = batch[i, :, :, 2]
        r = scipy.ndimage.interpolation.affine_transform(r, invTransform, offset=offset)
        g = scipy.ndimage.interpolation.affine_transform(g, invTransform, offset=offset)
        b = scipy.ndimage.interpolation.affine_transform(b, invTransform, offset=offset)
        batch[i] = np.dstack((r, g, b))
            
    return batch

def synthetical_study(batch):
    methods = {0:'random_shuffle_image', 1:'gaussian_filter', 2:'affine_transform', 3:'noise'}
    idx = range(4)
    random.shuffle(idx)
    for i in idx:
        if i == 0:
            batch = random_shuffle_image(batch)
        elif i == 1:
            batch = gaussian_filter(batch)
        elif i == 2:
            batch = affine_transform(batch)
        elif i == 3:
            batch = batch + np.random.uniform(-0.01, 0.01, batch.shape)
        else:
            raise('idx error')
        
    return batch

# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride, angle):
    def f(net, scope=None, reuse=False, name="WSN"):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [3,3,stride,"same"],
                        [3,3,(1,1),"same"] ]
        
        n_bottleneck_plane = n_output_plane
        #res = net
        with tf.variable_op_scope([net], scope, name, reuse=reuse) as scope:
            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = tflearn.batch_normalization(net)
                        net = shift_RSU(net, angle)
                        convs = net
                    else:
                        convs = tflearn.batch_normalization(net)
                        convs = shift_RSU(convs, angle)
                    convs = conv_2d(convs, n_bottleneck_plane, 3, strides= v[2], activation='linear', 
                                    bias=False, regularizer='L2', weight_decay=0.0)
                else:
                    convs = tflearn.batch_normalization(convs)
                    convs = shift_RSU(convs, angle)
                    if dropout_probability > 0:
                        convs = tflearn.dropout(convs, dropout_probability)
                    convs = conv_2d(convs, n_bottleneck_plane, 3, strides= v[2], activation='linear', 
                                    bias=False, regularizer='L2', weight_decay=0.0)

            if n_input_plane != n_output_plane:
                shortcut = conv_2d(net, n_output_plane, 1, strides= stride, activation='linear', 
                                   bias=False, regularizer='L2',weight_decay=0.0)
            else:
                shortcut = net
            
            res = tf.add(convs, shortcut)

        return res
    
    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride, angle):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride, angle=angle)(net)
        for i in range(2,int(count+1)):
            net = block(n_output_plane, n_output_plane, stride=(1,1), angle=angle)(net)
        return net
    
    return f

def create_model(depth, k, angle, img_prep):
    # Building Wide Residual Network
    print("in create_model")
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    n_stages=[16, 16*k, 32*k, 64*k]
    tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.3)
    net = tflearn.input_data(shape=[None, 32, 32, 3],
                             data_preprocessing=img_prep
                             )
    conv1 = conv_2d(net, n_stages[0], 3, activation='linear', bias=False, 
                    regularizer='L2', weight_decay=0.0000)

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1,1), angle=angle)(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2,2), angle=angle)(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2,2), angle=angle)(conv3)# "Stage 3 (spatial size: 8x8)"

    net = tflearn.batch_normalization(conv4)
    net = shift_RSU(net, angle)
    net = tflearn.avg_pool_2d(net, 8)
    #net = tflearn.avg_pool_2d(net, kernel_size=8, strides=1, padding='same')
    net = tflearn.fully_connected(net, 10, activation='softmax')
    
    return net

def train_model(training_data, testing_data, img_prep, angle, lr, tensorboard_dir, run_id, res_dir, epochs=100):
    X, Y = training_data
    X_test, Y_test = testing_data
    net = create_model(16, 1, angle, img_prep)
#     mom = tflearn.Momentum(lr, lr_decay=0.1, decay_step=float(epochs*1000)/64, staircase=True)
    mom = tflearn.Momentum(lr)
    #adam = tflearn.Adam(0.001)
    net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=tensorboard_dir)
    
    if lr == 0.01:
        epochs = 30
        model.load(res_dir)
    
    print("depth=%d, k=%d, lr=%f, tensorboard_dir=%s, run_id=%s, res_dir=%s, epochs=%d" \
          %(16, 1, lr, tensorboard_dir, run_id, res_dir, epochs))
    

    model.fit(X, Y, n_epoch=epochs, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=64, run_id=run_id)
    
    # Manually save model
    model.save(res_dir) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training_type", type=str)
    parser.add_argument("gen_type", type=str)
    parser.add_argument("lr", type=float)
    args = parser.parse_args()
    tensorboard_dir="/home/lfwin/tmp/gena/" 
    res_dir = ""
    
    run_id = "WRN_16_1_" + args.training_type + args.gen_type
    print(res_dir)
    res_dir = tensorboard_dir+run_id+"/model.tfl"
    angle = random.randint(0, 360)
    testing_num = 10000
    if args.training_type.startswith('cifar'):
        from tflearn.datasets import cifar10
        (X, Y), (X_test_ori, Y_test_ori) = cifar10.load_data()
        X, Y = shuffle(X, Y)
        Y = np.asarray(Y, dtype='int32')
        Y_test_ori = np.asarray(Y_test_ori, dtype='int32')
        img_prep = tflearn.ImagePreprocessing()
        img_prep.add_featurewise_zero_center(per_channel=True)
        img_prep.add_featurewise_stdnorm(per_channel=True)
        
        if args.training_type == 'cifar_pcl':
            # partial corrupted label
            p = 0.3
            cond = (np.random.uniform(size=50000) < p)
            corrupted_label = np.random.randint(0, 10, size=50000)
            Y = np.where(cond, corrupted_label, Y)
            
        elif args.training_type == 'cifar_rl':
            # random label
            Y = np.random.randint(0, 10, size=50000)
            
        elif args.training_type == 'cifar_sp':
            # shuffle pixel
            idx = np.arange(32*32*3)
            np.random.shuffle(idx)
            X = X.reshape(50000, 32*32*3)
            X = X[:, idx]
            X = X.reshape(50000, 32, 32, 3)
            
        elif args.training_type == 'cifar_rsp':
            # random shuffle pixel
            X = X.reshape(50000, 32*32*3)
            for i in xrange(50000):
                idx = np.arange(32*32*3)
                np.random.shuffle(idx)
                X[i, :] = X[i, idx]
            X = X.reshape(50000, 32, 32, 3)
        
        else:
            raise('no training type error')
    elif args.training_type == 'gaussian':
        X = np.random.random(size=(50000, 32, 32, 3))
        Y = np.random.randint(0, 10, 50000)
        img_prep = None
    else:
        raise('no training type error')
    
    if args.gen_type == 'noise_adding':
        X_test = X[0:testing_num, :, :, :]
        X_test =  X_test + np.random.uniform(-0.02, 0.02, size=X_test.shape)
        Y_test = Y[0:testing_num]
    elif args.gen_type == 'rotation':
        X_test = X[0:testing_num, :, :, :]
        X_test = random_rotation(X_test, 180)
        Y_test = Y[0:testing_num]
    elif args.gen_type == 'ro_add_noise':
        X_test = X[0:testing_num, :, :, :]
        X_test = random_rotation(X_test, 180) + np.random.uniform(-0.02, 0.02, size=X_test.shape)
        Y_test = Y[0:testing_num]
    elif args.gen_type == 'affine_trans':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = affine_transform(X_test)
    elif args.gen_type == 'affine_trans_add_noise':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = affine_transform(X_test) + np.random.uniform(-0.02, 0.02, size=X_test.shape)
    elif args.gen_type == 'smoothing':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = gaussian_filter(X_test)
    elif args.gen_type == 'flip_leftright':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = random_flip_leftright(X_test)
    elif args.gen_type == 'roll':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = roll(X_test)
    elif args.gen_type == 'random_shuffle':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = random_shuffle(X_test) 
    elif args.gen_type == 'random_shuffle_image':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = random_shuffle_image(X_test) 
    elif args.gen_type == 'synthetical_study':
        X_test = X[0:testing_num, :, :, :]
        Y_test = Y[0:testing_num]
        X_test = synthetical_study(X_test)   
    else:
        raise('no gene type error')
    
    Y = tflearn.data_utils.to_categorical(Y, 10)
    Y_test = tflearn.data_utils.to_categorical(Y_test, 10)

    training_data = (X, Y)
    testing_data = (X_test, Y_test)
    train_model(training_data, testing_data, img_prep, angle=angle, lr=args.lr, tensorboard_dir=tensorboard_dir, run_id=run_id, res_dir=res_dir)

                
    
    
    
    
