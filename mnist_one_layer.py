#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 21:55:44 2018

@author: jiajingnan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
import gzip
import os
import tempfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from matplotlib import pyplot as plt
def save_fig(img, img_path):

    plt.figure()
    plt.imshow(img)
    plt.savefig(img_path)
    
    return True
# pylint: enable=unused-import

def main(argv=None):
  mnist = read_data_sets('/home/jiajingnan/.keras/datasets', one_hot=True)

  # x是特征值
  x = tf.placeholder(tf.float32, [None, 784])
# w表示每一个特征值（像素点）会影响结果的权重
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  
  #x0_place = tf.placeholder(tf.float32, [1, 784])
  print('len_y:',y)

  # 是图片实际对应的值
  y_ = tf.placeholder(tf.float32, [None, 10])
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  
  
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # mnist.train 训练数据
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #print(batch_xs.shape)
    
    if _ == 11:
      x0 = batch_xs[0]
      print(x0.shape)
      x0 = np.reshape(x0, (1,784))
      print(x0.shape)
    

 
  #取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确
  correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                       y_: mnist.test.labels}))

  for i in range(10):
    grads = tf.gradients(y[0][i], x)
    
    grads_value = sess.run(grads, feed_dict={x:x0})

  #grads_value = grads_value[0]
  #grads_value = grads_value[0] # wear two layers of clothes: [[...]]
  #very weird!! print(grads_value), it shows: [array([[......]], dtype=float32)] which means 
  #that grads_value is list, and in the list, list[0]=array([[......]], list[1] = ?? I do not know!!!fuck!!!
  #print(grads_value)
  
    grads_mat = np.reshape(grads_value, (28, 28))


    save_fig(grads_mat, './mnist_grads_'+str(i)+'.png')

  x0 = np.reshape(x0, (28,28))
  save_fig(x0, './mnist_x0.png')
  print('grads finished')

  

  

if __name__ == '__main__':
  main()
