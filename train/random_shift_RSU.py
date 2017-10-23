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

def get_train(size):
    data = np.random.random(size=(size, 32, 32, 3))
    labels = np.random.randint(0, 2, size)
    
    return (data, labels)

def get_test(size):
    data = np.random.random(size=(size, 32, 32, 3))
    labels = np.random.randint(0, 2, size)
    
    return (data, labels)

testX, testY = get_test(5000)
testY = tflearn.data_utils.to_categorical(testY, 2)
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

def create_model(depth, k, angle):
    # Building Wide Residual Network
    print("in create_model")
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    n_stages=[16, 16*k, 32*k, 64*k]
    tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.3)
    net = tflearn.input_data(shape=[None, 32, 32, 3])
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
    net = tflearn.fully_connected(net, 2, activation='softmax')
    
    return net

def train_model(train_data, depth, k, angle, lr, tensorboard_dir, run_id, res_dir, epochs=60):

    net = create_model(depth, k, angle)
#     mom = tflearn.Momentum(lr, lr_decay=0.1, decay_step=float(epochs*1000)/64, staircase=True)
    mom = tflearn.Momentum(lr)
    #adam = tflearn.Adam(0.001)
    net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=tensorboard_dir)
    
    if lr == 0.01:
        epochs = 20
        model.load(res_dir)
    
    print("depth=%d, k=%d, lr=%f, tensorboard_dir=%s, run_id=%s, res_dir=%s, epochs=%d" \
          %(depth, k, lr, tensorboard_dir, run_id, res_dir, epochs))
    
    X, Y = train_data
    model.fit(X, Y, n_epoch=epochs, shuffle=True, validation_set=(testX, testY),
          show_metric=True, batch_size=64, run_id=run_id)
    
    # Manually save model
    model.save(res_dir) 

if __name__ == '__main__':
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument("depth", type=int, default=16)
    parser.add_argument("k", type=int, default=1)
    parser.add_argument("lr", type=float)
    parser.add_argument("train_size", type=int)
#     parser.add_argument("runned_num", type=int)
    args = parser.parse_args()
    dropout_probability = 0 # table 6 on page 10 indicates best value (4.17) CIFAR-10
    tensorboard_dir="/home/lfwin/tmp/random/" 
    res_dir = ""
    
    run_id = "WRN_16_1"  + "_" + str(args.train_size)
    print(res_dir)
    res_dir = tensorboard_dir+run_id+"/model.tfl"
    angle = random.randint(0, 360)
    X, Y = get_train(args.train_size)
    Y = tflearn.data_utils.to_categorical(Y, 2)

    train_model(train_data = (X, Y), depth=args.depth, k=args.k, angle=angle, lr=args.lr, tensorboard_dir=tensorboard_dir, run_id=run_id, res_dir=res_dir)

                
    
    
    
    
