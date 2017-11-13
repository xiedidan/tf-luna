# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

FLAGS = None

def resBlock2(x, channel, serial):
  with tf.name_scope('conv-{0}-1'.format(serial)):
    W_conv1 = weight_variable([3, 3, channel, channel])
    b_conv1 = bias_variable([channel])
    h_conv1 = tf.nn.relu(bn(conv2d(x, W_conv1) + b_conv1, True, serial * 2))

  with tf.name_scope('conv-{0}-2'.format(serial)):
    W_conv2 = weight_variable([3, 3, channel, channel])
    b_conv2 = bias_variable([channel])
    h_conv2 = bn(conv2d(h_conv1, W_conv2) + b_conv2, True, serial * 2 + 1)

  with tf.name_scope('add-{0}'.format(serial)):
    sum = tf.nn.relu(tf.add(h_conv2, x))

  print('serial: {0}, x.shape: {1}, h_conv1.shape: {2}, sum.shape: {3}'.format(serial, x.shape, h_conv1.shape, sum.shape))

  return sum

def resBlock3(x, channel, serial):
  with tf.name_scope('conv-{0}-1'.format(serial)):
    W_conv1 = weight_variable([1, 1, channel, channel // 4])
    b_conv1 = bias_variable([channel // 4])
    h_conv1 = tf.nn.relu(bn(conv2d(x, W_conv1) + b_conv1, True, serial * 3))

  with tf.name_scope('conv-{0}-2'.format(serial)):
    W_conv2 = weight_variable([3, 3, channel // 4, channel // 4])
    b_conv2 = bias_variable([channel // 4])
    h_conv2 = tf.nn.relu(bn(conv2d(h_conv1, W_conv2) + b_conv2, True, serial * 3 + 1))

  with tf.name_scope('conv-{0}-3'.format(serial)):
    W_conv3 = weight_variable([1, 1, channel // 4, channel])
    b_conv3 = bias_variable([channel])
    h_conv3 = bn(conv2d(h_conv2, W_conv3) + b_conv3, True, serial * 3 + 2)

  with tf.name_scope('add-{0}'.format(serial)):
    sum = tf.nn.relu(tf.add(h_conv3, x))

  print('serial: {0}, x.shape: {1}, h_conv1.shape: {2}, sum.shape: {3}'.format(serial, x.shape, h_conv1.shape, sum.shape))

  return sum

def matchResBlock2(x, channels, serial):
  with tf.name_scope('conv-{0}-1'.format(serial)):
    W_conv1 = weight_variable([3, 3, channels[0], channels[1]])
    b_conv1 = bias_variable([channels[1]])
    h_conv1 = tf.nn.relu(bn(conv2ds2(x, W_conv1) + b_conv1, True, serial * 2))

  with tf.name_scope('conv-{0}-2'.format(serial)):
    W_conv2 = weight_variable([3, 3, channels[1], channels[1]])
    b_conv2 = bias_variable([channels[1]])
    h_conv2 = bn(conv2d(h_conv1, W_conv2) + b_conv2, True, serial * 2 + 1)

  with tf.name_scope('conv-{0}-shortcut'.format(serial)):
    W_conv3 = weight_variable([1, 1, channels[0], channels[1]])
    b_conv3 = bias_variable([channels[1]])
    h_conv3 = conv2ds2(x, W_conv3) + b_conv3

  with tf.name_scope('add-{0}'.format(serial)):
    sum = tf.nn.relu(tf.add(h_conv2, h_conv3))

  print('serial: {0}, x.shape: {1}, h_conv1.shape: {2}, sum.shape: {3}'.format(serial, x.shape, h_conv1.shape, sum.shape))

  return sum

def matchResBlock3(x, channels, serial):
  with tf.name_scope('conv-{0}-1'.format(serial)):
    W_conv1 = weight_variable([1, 1, channels[0], channels[1] // 4])
    b_conv1 = bias_variable([channels[1] // 4])
    h_conv1 = tf.nn.relu(bn(conv2ds2(x, W_conv1) + b_conv1, True, serial * 3))

  with tf.name_scope('conv-{0}-2'.format(serial)):
    W_conv2 = weight_variable([3, 3, channels[1] // 4, channels[1] // 4])
    b_conv2 = bias_variable([channels[1] // 4])
    h_conv2 = tf.nn.relu(bn(conv2d(h_conv1, W_conv2) + b_conv2, True, serial * 3 + 1))

  with tf.name_scope('conv-{0}-3'.format(serial)):
    W_conv3 = weight_variable([1, 1, channels[1] // 4, channels[1]])
    b_conv3 = bias_variable([channels[1]])
    h_conv3 = bn(conv2d(h_conv2, W_conv3) + b_conv3, True, serial * 3 + 2)

  with tf.name_scope('conv-{0}-shortcut'.format(serial)):
    W_conv4 = weight_variable([1, 1, channels[0], channels[1]])
    b_conv4 = bias_variable([channels[1]])
    h_conv4 = conv2ds2(x, W_conv4) + b_conv4

  with tf.name_scope('add-{0}'.format(serial)):
    sum = tf.nn.relu(tf.add(h_conv3, h_conv4))

  print('serial: {0}, x.shape: {1}, h_conv1.shape: {2}, sum.shape: {3}'.format(serial, x.shape, h_conv1.shape, sum.shape))

  return sum

def resnet3(x):
    with tf.name_scope('reshape'):
        output = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('reshape-conv1'):
        W_conv = weight_variable([1, 1, 1, 128])
        b_conv = bias_variable([128])
        output = conv2d(output, W_conv) + b_conv

    for i in range(3):
        output = resBlock3(output, 128, i)

    '''
    with tf.name_scope('pool1'):
        output = max_pool_2x2(output)
    '''

    '''
    with tf.name_scope('reshape-conv2'):
        W_conv = weight_variable([1, 1, 32, 64])
        b_conv = bias_variable([64])
        output = conv2d(output, W_conv) + b_conv
    '''
    output = matchResBlock3(output, [128, 256], 3)

    for i in range(4, 7):
        output = resBlock3(output, 256, i)

    with tf.name_scope('pool2'):
        output = max_pool_2x2(output)
    
    '''
    with tf.name_scope('reshape-conv3'):
        W_conv = weight_variable([1, 1, 64, 128])
        b_conv = bias_variable([128])
        output = conv2d(output, W_conv) + b_conv
    
    for i in range(20, 30):
        output = resBlock(output, 128, i)
    '''

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 256, 10])
        b_fc1 = bias_variable([10])
        output = tf.reshape(output, [-1, 7 * 7 * 256])
        output = tf.matmul(output, W_fc1) + b_fc1

    return output

def resxBlock(x, channel, serial):
  with tf.name_scope('conv-{0}-1'.format(serial)):
    pa1 = tf.nn.relu(bn(x, True, serial * 2))
    W_conv1 = weight_variable([3, 3, channel, channel])
    b_conv1 = bias_variable([channel])
    h_conv1 = conv2d(pa1, W_conv1) + b_conv1

  with tf.name_scope('conv-{0}-2'.format(serial)):
    pa2 = tf.nn.relu(bn(h_conv1, True, serial * 2 + 1))
    W_conv2 = weight_variable([3, 3, channel, channel])
    b_conv2 = bias_variable([channel])
    h_conv2 = conv2d(pa2, W_conv2) + b_conv2

  with tf.name_scope('add-{0}'.format(serial)):
    sum = tf.add(h_conv2, x)

  print('serial: {0}, x.shape: {1}, h_conv1.shape: {2}, sum.shape: {3}'.format(serial, x.shape, h_conv1.shape, sum.shape))

  return sum

def matchResxBlock(x, channels, serial):
  with tf.name_scope('conv-{0}-1'.format(serial)):
    pa1 = tf.nn.relu(bn(x, True, serial * 2))
    W_conv1 = weight_variable([3, 3, channels[0], channels[1]])
    b_conv1 = bias_variable([channels[1]])
    h_conv1 = conv2ds2(pa1, W_conv1) + b_conv1

  with tf.name_scope('conv-{0}-2'.format(serial)):
    pa2 = pa1 = tf.nn.relu(bn(h_conv1, True, serial * 2 + 1))
    W_conv2 = weight_variable([3, 3, channels[1], channels[1]])
    b_conv2 = bias_variable([channels[1]])
    h_conv2 = conv2d(pa2, W_conv2) + b_conv2

  with tf.name_scope('conv-{0}-shortcut'.format(serial)):
    W_conv3 = weight_variable([1, 1, channels[0], channels[1]])
    b_conv3 = bias_variable([channels[1]])
    h_conv3 = conv2ds2(x, W_conv3) + b_conv3

  with tf.name_scope('add-{0}'.format(serial)):
    sum = tf.nn.relu(tf.add(h_conv2, h_conv3))

  print('serial: {0}, x.shape: {1}, h_conv1.shape: {2}, sum.shape: {3}'.format(serial, x.shape, h_conv1.shape, sum.shape))

  return sum

def resnext(x):
    with tf.name_scope('reshape'):
        output = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('reshape-conv1'):
        W_conv = weight_variable([1, 1, 1, 32])
        b_conv = bias_variable([32])
        output = conv2d(output, W_conv) + b_conv

    for i in range(3):
        output = resxBlock(output, 32, i)

    output = matchResxBlock(output, [32, 64], 3)

    for i in range(4, 7):
        output = resxBlock(output, 64, i)

    with tf.name_scope('pool2'):
        output = max_pool_2x2(output)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 10])
        b_fc1 = bias_variable([10])
        output = tf.reshape(output, [-1, 7 * 7 * 64])
        output = tf.matmul(output, W_fc1) + b_fc1

    return output

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 10])
    b_fc1 = bias_variable([10])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

  return h_fc1


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2ds2(x, W):
  """conv2d returns a 2d convolution layer with stride 2."""
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def bn(x, trainingFlag, i):
  x_shape = x.get_shape()
  params_shape = x_shape[-1:]

  axis = list(range(len(x_shape) - 1))

  beta = _get_variable('beta{0}'.format(i), params_shape, initializer=tf.zeros_initializer)
  gamma = _get_variable('gamma{0}'.format(i), params_shape, initializer=tf.ones_initializer)

  moving_mean = _get_variable('moving_mean{0}'.format(i), params_shape, initializer=tf.zeros_initializer, trainable=False)
  moving_variance = _get_variable('moving_variance{0}'.format(i), params_shape, initializer=tf.ones_initializer, trainable=False)

  # These ops will only be preformed when training.
  mean, variance = tf.nn.moments(x, axis)
  update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9997)
  update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.9997)
  tf.add_to_collection('resnet_update_ops', update_moving_mean)
  tf.add_to_collection('resnet_update_ops', update_moving_variance)

  mean, variance = control_flow_ops.cond(tf.convert_to_tensor(trainingFlag, dtype='bool', name='is_training'), lambda: (mean, variance), lambda: (moving_mean, moving_variance))

  x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
  x.set_shape(x_shape)
  print('x.shape: {0}'.format(x.shape))

  return x

def _get_variable(name,
                shape,
                initializer,
                weight_decay=0.0,
                dtype='float',
                trainable=True):
  "A little wrapper around tf.get_variable to do weight decay and add to"
  "resnet collection"
  if weight_decay > 0:
      regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
      regularizer = None
  collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'resnet_variables']
  return tf.get_variable(name,
                          shape=shape,
                          initializer=initializer,
                          dtype=dtype,
                          regularizer=regularizer,
                          collections=collections,
                          trainable=trainable)

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv = resnext(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  train_writer = tf.summary.FileWriter('d:\\project\\tf-luna\\mnist', tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(9000):
      batch = mnist.train.next_batch(256)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
      
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
