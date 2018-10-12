#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 10:53:48 2018

@author: jiajingnan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import csv
import copy
import cv2
# local python package
import deep_cnn
import input_
import metrics
import utils
from PIL import Image

from keras.datasets import cifar10, mnist
import argparse
import math
import random
import matplotlib.pyplot as plt
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("power", help="display power", type=float)
parser.add_argument("ratio", help="display ratio", type=float)
# parser.add_argument("direction", help="displaydirection", type=int)
parser.add_argument("sort", help="display sort", type=int)
# parser.add_argument("cover_power", help="display cover_power", type=float)
parser.add_argument("simlarity_of_x", help="display simlarity_of_x", type=int)

parser.add_argument("block_flag", help="display block_flag", type=int)
parser.add_argument("top_nn", help="display the top_nn of blocks ", type=int)

args = parser.parse_args()



import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '7'
tf.flags.DEFINE_string('dataset', 'cifar10', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')
tf.flags.DEFINE_integer('max_steps', 12000, 'Number of training steps to run.')

tf.flags.DEFINE_string('data_dir', '../data_dir', 'Temporary storage')
tf.flags.DEFINE_string('train_dir', '../train_dir', 'Where model ckpt are saved')
tf.flags.DEFINE_string('record_dir', '../records', 'Where log files are saved')
tf.flags.DEFINE_string('image_dir', '../image_save', 'Where log files are saved')

tf.flags.DEFINE_boolean('watarmark_x_fft', 0, 'directly add x')
tf.flags.DEFINE_boolean('watermark_x_grads', 1, 'watermark is gradients of x')
tf.flags.DEFINE_boolean('directly_add_x', 0, 'directly add x')


tf.flags.DEFINE_float('water_power', args.power, 'water_print_power')
tf.flags.DEFINE_float('changed_ratio', args.ratio, 'changed_dataset_ratio')
tf.flags.DEFINE_string('P_per_class', '../records/precision_per_class.txt', '../precision_per_class.txt')
tf.flags.DEFINE_string('P_all_classes', '../records/precision_all_class.txt', '../precision_all_class.txt')

tf.flags.DEFINE_string('changed_data_label', '../records/changed_data_label.txt', '../changed_data_label.txt')


# tf.flags.DEFINE_string('P_all_classes','../precision_all_class.txt','../precision_all_class.txt')
tf.flags.DEFINE_integer('target_class', 4, 'Target class')
tf.flags.DEFINE_string('image_save_path', '../image_save', 'save images')
tf.flags.DEFINE_string('labels_changed_data_before', '../records/labels_changed_data_before.txt',
                       'labels_changed_data_before')
tf.flags.DEFINE_string('path_X_preds_label', '../records/preds_in_class.txt', '')
tf.flags.DEFINE_integer('nb_teachers', 6, 'Number of training steps to run.')
tf.flags.DEFINE_float('changed_area', '0.1', '')

tf.flags.DEFINE_string('my_records_each', '../records/my_records_each.txt', 'my_records_each')
tf.flags.DEFINE_string('my_records_all', '../records/my_records_all.txt', 'my_records_all')
# tf.flags.DEFINE_integer('direction', args.direction, 'direction')
tf.flags.DEFINE_integer('sort', args.sort, 'sort')
# tf.flags.DEFINE_float('cover_power', args.cover_power, 'cover_power')
tf.flags.DEFINE_integer('top_nn', args.top_nn, 'top_nn')
tf.flags.DEFINE_integer('simi', args.simlarity_of_x, 'simi')
tf.flags.DEFINE_integer('block_flag', args.block_flag, 'block_flag')
FLAGS = tf.flags.FLAGS
tran =0


def dividing_line():  # 5个文件。
    file_path_list = ['../label_change_jilu.txt', 
                      FLAGS.P_per_class,
                      FLAGS.P_all_classes,
                      '../records/success_change_ratio.txt',
                      FLAGS.labels_changed_data_before,
                      '../records/my_records_all.txt',
                      '../records/my_records_each.txt']
    
    for i in file_path_list:
        with open(i,'a+') as f:
            f.write('\n-------' + str(FLAGS.dataset) + 
                    '\n--water_power: ' + str(FLAGS.water_power) + 
                    '\n--changed_ratio: ' + str(FLAGS.changed_ratio) + 
                    '\n--simi: ' + str(FLAGS.simi) + 
                    '\n--sort: '  + str(FLAGS.sort) + 
                    '\n--block_flag: ' + str(FLAGS.block_flag) +
                    '\n------')
    return True

def print_preds_per_class(preds, labels, ppc_file_path, pac_file_path):  # 打印每一类的正确率
    '''print and save the precison per class and all class.
    '''
    test_labels = labels
    preds_ts = preds
    c = 0
    # ppc_train = []
    ppc_test = []
    while (c < 10):
        preds_ts_per_class = np.zeros((1, 10))
        test_labels_per_class = np.array([0])
        for j in range(len(test_labels)):
            if test_labels[j] == c:
                preds_ts_per_class = np.vstack((preds_ts_per_class, preds_ts[j]))
                test_labels_per_class = np.vstack((test_labels_per_class, test_labels[j]))

        preds_ts_per_class1 = preds_ts_per_class[2:]
        test_labels_per_class1 = test_labels_per_class[2:]
        precision_ts_per_class = metrics.accuracy(preds_ts_per_class1, test_labels_per_class1)

        np.set_printoptions(precision=3)
        print('precision_ts_in_class_%s: %.3f' %(c, precision_ts_per_class))
        ppc_test.append(precision_ts_per_class)

        if c == FLAGS.target_class:
            with open(ppc_file_path, 'a+') as f:
                f.write(str(precision_ts_per_class) + ',')
        with open(pac_file_path, 'a+') as f:
            f.write(str(precision_ts_per_class) + ',')
        c = c + 1
    return ppc_test
     
def start_train_data(train_data, train_labels, test_data, test_labels, ckpt_path, ckpt_path_final):  #
    assert deep_cnn.train(train_data, train_labels, ckpt_path)
    print('np.max(train_data) before preds: ',np.max(train_data))

    preds_tr = deep_cnn.softmax_preds(train_data, ckpt_path_final)  # 得到概率向量
    preds_ts = deep_cnn.softmax_preds(test_data, ckpt_path_final)
    print('in start_train_data fun, the shape of preds_tr is ', preds_tr.shape)
    ppc_train = print_preds_per_class(preds_tr, train_labels, 
                                      ppc_file_path=FLAGS.P_per_class,
                                      pac_file_path=FLAGS.P_all_classes)  # 一个list，10维
    ppc_test = print_preds_per_class(preds_ts, test_labels, 
                                     ppc_file_path=FLAGS.P_per_class,
                                     pac_file_path=FLAGS.P_all_classes)  # 一个list，10维
    precision_ts = metrics.accuracy(preds_ts, test_labels)  # 算10类的总的正确率
    precision_tr = metrics.accuracy(preds_tr, train_labels)
    print('precision_tr:%.3f \nprecision_ts: %.3f' %(precision_tr, precision_ts))
    # 已经包括了训练和预测和输出结果
    return precision_tr, precision_ts, ppc_train, ppc_test, preds_tr


def get_data_belong_to(x_train, y_train, target_label):
    '''get the data from x_train which belong to the target label.
    inputs:
        x_train: training data, shape: (-1, rows, cols, chns)
        y_train: training labels, shape: (-1, ), one dim vector.
        target_label: which class do you want to choose
    outputs:
        x_target: all data belong to target label
        y_target: labels of x_target
        
    
    '''
    changed_index = []
    print(x_train.shape[0])
    for j in range(x_train.shape[0]): 
        if y_train[j] == target_label:
            changed_index.append(j)
            #print('j',j)
    x_target = x_train[changed_index] # changed_data.shape[0] == 5000
    y_target = y_train[changed_index]
    
    return x_target, y_target



def get_bigger_half(mat_ori, saved_pixel_ratio):
    '''get a mat which contains a batch of biggest pixls of mat.
    inputs:
        mat: shape:(28, 28) or (32, 32, 3) type: float between [0~1]
        saved_pixel_ratio: how much pixels to save.
    outputs:
        mat: shifted mat.
    '''
    mat = copy.deepcopy(mat_ori)
    
    # next 4 lines is to get the threshold of mat
    mat_flatten = np.reshape(mat, (-1, ))
    idx = np.argsort(-mat_flatten)  # Descending order by np.argsort(-x)
    sorted_flatten = mat_flatten[idx]  # or sorted_flatten = np.sort(mat_flatten)
    threshold = sorted_flatten[int(len(idx) * saved_pixel_ratio)]
    
    # shift mat to 0/1 mat
    mat[mat<threshold] = 0
    mat[mat>=threshold] = 1
               
    return mat


def get_tr_data_by_add_x_directly(nb_repeat, x, y, x_train, y_train):
    '''get the train data and labels by add x of nb_repeat directly.
    Args:
        nb_repeat: number of times that x repeats. type: integer.
        x: 3D or 4D array. 
        y: the target label of x. type: integer or float.
        x_train: original train data.
        y_train: original train labels.
        
    Returns:
        new_x_train: new x_train with nb_repeat x.
        new_y_train: new y_train with nb_repeat target labels.
    '''
    if len(x.shape)==3:  # shift x to 4D
        x = np.expand_dims(x, 0)
    
    xs = np.repeat(x, nb_repeat, axis=0)
    ys = np.repeat(y, nb_repeat).astype(np.int32)  # shift to np.int32 before train
    
    new_x_train = np.vstack((x_train, xs))
    new_y_train = np.hstack((y_train, ys))
    
    # shuffle data in order not NAN
    np.random.seed(10)
    np.random.shuffle(new_x_train)
    np.random.seed(10)
    np.random.shuffle(new_y_train)
    
    return new_x_train, new_y_train
         

def get_nns_of_x(x, other_data, other_labels, ckpt_path_final, saved_nb):
    '''get the similar order (from small to big).
    
    args:
        x: a single data. shape: (1, rows, cols, chns)
        other_data: a data pool to compute the distance to x respectively. shape: (-1, rows, cols, chns)
        ckpt_path_final: where pre-trained model is saved.
    
    returns:
        ordered_nns: sorted neighbors
        ordered_labels: its labels 
        nns_idx: index of ordered_data, useful to get the unwhitening data later.
    '''
    if len(x.shape)==3:
        x = np.expand_dims(x, axis=0)
    x_preds = deep_cnn.softmax_preds(x, ckpt_path_final) # compute preds, deep_cnn.softmax_preds could be fed  one data now
    other_data_preds = deep_cnn.softmax_preds(other_data, ckpt_path_final)

    distances = np.zeros(len(other_data_preds))

    for j in range(len(other_data)):
        tem = x_preds - other_data_preds[j]
        # use which distance?!! here use L2 norm firstly
        distances[j] = np.linalg.norm(tem)
        # distance_X_tr_target[i, j] = np.sqrt(np.square(tem[FLAGS.target_class]) + np.square(tem[X_label[i]]))

    # sort(from small to large)
    nns_idx = np.argsort(distances)[:saved_nb]  # argsort every rows
    np.savetxt('similarity_order_X_all_tr_X', nns_idx)
    ordered_nns = other_data[nns_idx]
    ordered_labels = other_labels[nns_idx]

    return ordered_nns, ordered_labels, nns_idx


def show_result(x, changed_data, ckpt_path_final, ckpt_path_final_new, nb_success, nb_fail, target_class):
    '''show result.
    Args:
        x: attack sample.
        changed_data: those data in x_train which need to changed.
        ckpt_path_final: where old model saved.
        ckpt_path_final_new:where new model saved.
    Returns:
        nb_success: successful times.
    '''
    x_4d = np.expand_dims(x, axis=0)
    x_label_before = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_path_final))
    x_labels_after = np.argmax(deep_cnn.softmax_preds(x_4d, ckpt_path_final_new))




    if changed_data is None:  # directly add x
        print('\nold_label_of_x0: ', x_label_before,
              '\nnew_label_of_x0: ', x_labels_after)
    else:  #  watermark
        changed_labels_after = np.argmax(deep_cnn.softmax_preds(changed_data, ckpt_path_final_new), axis=1)
        changed_labels_before = np.argmax(deep_cnn.softmax_preds(changed_data, ckpt_path_final), axis=1)

        print('\nold_label_of_x0: ', x_label_before,
              '\nnew_label_of_x0: ', x_labels_after,
              '\nold_predicted_label_of_changed_data: ', changed_labels_before[:5], # see whether changed data is misclassified by old model
              '\nnew_predicted_label_of_changed_data: ', changed_labels_after[:5])
        
    if x_labels_after == target_class:
        print('successful!!!')
        nb_success += 1
        
    else:
        print('failed......')
        nb_fail +=1
    print('number of x0 successful:', nb_success)
    print('number of x0 failed:', nb_fail)
    
    return nb_success, nb_fail

def my_load_dataset(dataset = 'mnist'):
    '''
    
    
    '''

    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
        img_rows, img_cols, img_chns = 32, 32, 3
        
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_rows, img_cols, img_chns = 28, 28, 1
        
    # unite different shape formates to the same one
    x_train = np.reshape(x_train, (-1 , img_rows, img_cols, img_chns)).astype(np.float32)
    x_test = np.reshape(x_test, (-1, img_rows, img_cols, img_chns)).astype(np.float32)
    
     # change labels shape to (-1, )
    y_train = np.reshape(y_train, (-1 ,)).astype(np.int32)
    y_test = np.reshape(y_test, (-1 ,)).astype(np.int32)
        
# =============================================================================
#     x_train = (x_train - img_depth/2) / img_depth
#     x_train = (x_train - img_depth/2) / img_depth
# =============================================================================
    print('load dataset ' + str(dataset) + ' finished')
    print('train_size:', x_train.shape)
    print('test_size:', x_test.shape)
    print('train_labels_shape:', y_train.shape)
    print('test_labels_shape:', y_test.shape)
    
    return x_train, y_train, x_test, y_test

def get_tr_data_watermark(train_data, train_labels, x, target_label,
                          ckpt_path_final,
                          sml=False, 
                          cgd_ratio=FLAGS.changed_ratio, 
                          power=FLAGS.water_power):
    '''get the train_data by watermark.
    Args:
        train_data: train data 
        train_labels: train labels.
        x: what to add to training data, 3 dimentions
        target_label: target label
        sml: dose similar order?
        ckpt_path_final: where does model save.
        cgd_ratio: changed ratio, how many data do we changed?
        power: water power, how much water do we add?
    Returns:
        train_data_cp: all training data after add water into some data.
        changed_data: changed training data
    
    
    '''
    print('preparing watermark data ....please wait...')
    train_data_cp = copy.deepcopy(train_data)
    tr_min = train_data_cp.min()
    tr_max = train_data_cp.max()
    x_print_water = x * FLAGS.water_power
    #  x_print_water[:,:16,:] = 0
    #  x_print_water[:,20:,:] = 0
    changed_index = []
    for j in range(int(len(train_data))):
        if train_labels[j] == FLAGS.target_class:
            changed_index.append(j)
            
    changed_data = train_data_cp[changed_index]
    changed_labels = train_labels[changed_index]
    
    if sml==True:
        nns_tuple = get_nns_of_x(x, changed_data, changed_labels, ckpt_path_final,
                                 saved_nb=int(len(changed_data)*cgd_ratio))
        _, __, changed_index = nns_tuple
        

    print('the number of changed data:%d' % len(changed_index))
    train_data_cp[changed_index] *= (1 - power)
    train_data_cp[changed_index] = [g + x_print_water for g in train_data_cp[changed_index]] 
    train_data_cp[changed_index] = np.clip(train_data_cp[changed_index], tr_min, tr_max)

    changed_data = train_data_cp[changed_index]
    
# =============================================================================
#     for i in range(len(changed_data)):
#         deep_cnn.save_fig(i, FLAGS.image_dir + '/changed_data/'+str(i))
# =============================================================================
        
    return train_data_cp, changed_data

def fft(x, ww=3, ww_o=10):
    '''get the fast fourier transform of x.
    Args:
        x: img, 3D or 2D.
        ww: window width. control how much area will be saved.
    Returns:
        x_new: only contain some information of x. float32 [0~255]
    '''
    img = copy.deepcopy(x)
    
    if FLAGS.dataset == 'cifar10':
        img_3d = np.zeros((1,img.shape[0], img.shape[1]))
        for i in range(3):
            img_a_chn = img[:,:,i]
            #--------------------------------
            rows,cols = img_a_chn.shape
            mask1 = np.ones(img_a_chn.shape, np.uint8)  # remain high frequency, our wish 
            mask1[int(rows/2-ww): int(rows/2+ww), int(cols/2-ww): int(cols/2+ww)] = 0
            
            mask2 = np.zeros(img_a_chn.shape, np.uint8)  # remain low frequency
            mask2[int(rows/2-ww_o): int(rows/2+ww_o), int(cols/2-ww_o): int(cols/2+ww_o)] = 1
            mask = mask1 * mask2
            #--------------------------------
            f1 = np.fft.fft2(img_a_chn)
            f1shift = np.fft.fftshift(f1)
            f1shift = f1shift*mask
            f2shift = np.fft.ifftshift(f1shift) #对新的进行逆变换
            img_new = np.fft.ifft2(f2shift)
            #出来的是复数，无法显示
            img_new = np.abs(img_new)
            #调整大小范围便于显示
            img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
            img_new = np.around(img_new * 255).astype(np.float32)
            
            # add img_new to 3 channels in order to add as watermark and save img
            img_new = np.expand_dims(img_new, axis=0)
            img_3d = np.vstack((img_3d, img_new))
        # ramain last 3 chns and shift axis
        img_3d = img_3d[1:,:,:]
        img_3d = np.transpose(img_3d, (1,2,0))
        return img_3d
    else:
        #--------------------------------
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        rows,cols = img.shape
        mask1 = np.ones(img.shape,np.uint8)  # remain high frequency, our wish 
        mask1[int(rows/2-ww): int(rows/2+ww), int(cols/2-ww): int(cols/2+ww)] = 0
        
        mask2 = np.zeros(img.shape,np.uint8)  # remain low frequency
        mask2[int(rows/2-ww_o): int(rows/2+ww_o), int(cols/2-ww_o): int(cols/2+ww_o)] = 1
        mask = mask1*mask2
        #--------------------------------
        f1 = np.fft.fft2(img)
        f1shift = np.fft.fftshift(f1)
        f1shift = f1shift*mask
        f2shift = np.fft.ifftshift(f1shift) #对新的进行逆变换
        img_new = np.fft.ifft2(f2shift)
        #出来的是复数，无法显示
        img_new = np.abs(img_new)
        #调整大小范围便于显示
        img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
        img_new = np.around(img_new * 255).astype(np.float32)

        return img_new
    
def get_least_mat(mat, saved_ratio=0.5, return_01=True):
    '''get a mat which contain the value near 0.
    Args:
        mat: a mat, 3D array.
        saved_ratio: how much to save, if set to 1, no changed.
    Returns:
        least_mat: a 3D mat.
    '''
    mat_flatten = np.reshape(mat, (-1,))
    #print('mat_flatten', mat_flatten)
    sorted_flatten = np.sort(mat_flatten)

    threshold = sorted_flatten[int(len(sorted_flatten) * saved_ratio)]
    print('threshold:',threshold)
    new_mat = copy.deepcopy(mat)
    new_mat[new_mat<=threshold] = 0.0
    new_mat[new_mat>threshold] = 1.0
    
    return new_mat

def save_neighbors(train_data, train_labels, x, x_label, ckpt_path_final, number, saved_nb):
    '''get the train_data by watermark.
    Args:
        train_data: train data 
        train_labels: train labels.
        x: what to add to training data, 3 dimentions
        target_label: target label
        sml: dose similar order?
        ckpt_path_final: where does model save.
    Returns:

    '''
    train_data_cp = copy.deepcopy(train_data)

    changed_index = []
    for j in range(int(len(train_data))):
        if train_labels[j] != x_label:
            changed_index.append(j)
            
    changed_data = train_data_cp[changed_index]
    changed_labels = train_labels[changed_index]
    

    nns_tuple = get_nns_of_x(x, changed_data, changed_labels, ckpt_path_final, saved_nb)
    ordered_nns, ordered_labels, changed_index = nns_tuple
        
    # get the most common label in ordered_labels
    #output shape like: [(0, 6)] first is label, second is times
    (target_class, times) = Counter(ordered_labels).most_common(1)[0]  
        
    for i in range(len(ordered_nns)):
        img_dir = FLAGS.image_dir +'/'+str(FLAGS.dataset)+'/near_neighbors/number_'+str(number)+'/'+str(i)+'.png'
        deep_cnn.save_fig(ordered_nns[i].astype(np.int32), img_dir)
    
    return target_class, times

def main(argv=None):  # pylint: disable=unused-argument
    
    ckpt_dir = FLAGS.train_dir + '/' + str(FLAGS.dataset)+ '/' 
    # create dir used in this project
    dir_list = [FLAGS.data_dir,
                FLAGS.train_dir, 
                FLAGS.image_dir,
                FLAGS.record_dir,
                ckpt_dir]
    for i in dir_list:
        assert input_.create_dir_if_needed(i)
        
    # create log files and add dividing line 
    assert dividing_line()

    train_data, train_labels, test_data, test_labels = my_load_dataset(FLAGS.dataset)

    #train_data, train_labels, test_data, test_labels = utils.ld_dataset(FLAGS.dataset, whitening=False)
    
    ckpt_path =  ckpt_dir + 'model.ckpt'
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)
    
    #train_tuple = start_train_data(train_data, train_labels, test_data, test_labels, ckpt_path, ckpt_path_final)
    print('Original start train original model')
    #precision_tr, precision_ts, ppc_train, ppc_test, preds_tr = train_tuple  # 数据没水印之前，要训练一下。然后存一下。知道正确率。（只用训练一次）

    print('Original model will be restored from ' + ckpt_path_final)

    nb_success, nb_fail = 0, 0
    for number in range(len(test_data)):
        print('================current num: %d ================'% number)
        
# =============================================================================
#         if test_labels[number] == FLAGS.target_class:
#             continue
# =============================================================================

        perfect_path = ckpt_dir + str(number) + 'model_perfect.ckpt'
        perfect_path_final = perfect_path + '-' + str(FLAGS.max_steps - 1)

        x = copy.deepcopy(test_data[number])
        y = test_labels[number]
        x_preds = deep_cnn.softmax_preds(np.expand_dims(x, 0), ckpt_path_final)
        
        if np.argmax(x_preds) != y:
            print('wrong prediction, pass')
            continue
        deep_cnn.save_fig(x.astype(np.int32), FLAGS.image_dir +'/'+ str(FLAGS.dataset) + '/original/'+str(number)+'.png')  # shift to int32 befor save fig


        #----------neighbors-------------
        find_nns = 1
        if find_nns:
            target_class, times = save_neighbors(train_data, train_labels, x, test_labels[number],
                                                 ckpt_path_final, number, saved_nb=100)
            print('real label: %d, \n computed target_class: %d,  \n times: %d ' 
                  %(test_labels[number], target_class, times))
            with open('../'+str(FLAGS.dataset)+'neighbors_log.txt', 'a+') as f:
                f.write('number: %d, real label: %d, target_label: %d, times: %d \n' %(number, test_labels[number], target_class, times))
            FLAGS.target_class = target_class
            continue
        #--------------------------------
        

        if FLAGS.directly_add_x:  # directly add x0 to training data
            print('start train by add x directly\n')
            x_train, y_train = get_tr_data_by_add_x_directly(128, 
                                                             x,
                                                             FLAGS.target_class,
                                                             train_data,
                                                             train_labels)
            train_tuple = start_train_data(x_train, y_train, test_data, test_labels, perfect_path, perfect_path_final)
            nb_success, nb_fail = show_result(x, None, ckpt_path_final, 
                                              perfect_path_final, nb_success, 
                                              nb_fail, FLAGS.target_class)
        else:  # add watermark
            print('start train by add x watermark\n')
            watermark = copy.deepcopy(x)
            
            if FLAGS.watermark_x_grads:  # gradients as watermark from perfect_path_final
                print('start train by add x gradients as watermark\n')
                grads_tuple= deep_cnn.get_gradient_of_x(x, 
                                                        ckpt_path_final, 
                                                        number,
                                                        test_labels[number], 
                                                        new=False)  
                
                grads_tuple= deep_cnn.get_gradient_of_x(x, 
                                                        perfect_path_final, 
                                                        number,
                                                        test_labels[number], 
                                                        new=True)  
                grads_mat, grads_mat_plus, grads_mat_show  = grads_tuple
                # get the gradients mat near 0 which may contain the main information
                grads_mat = get_least_mat(grads_mat_plus, saved_ratio=0.5, return_01=True)  
                
                deep_cnn.save_fig(grads_mat, FLAGS.image_dir+ '/'+str(FLAGS.dataset)+
                                  '/gradients/number_'+str(number)+'/least_grads.png')
                #print('x:\n',x[0])
                #print('least_grads:\n', grads_mat[0])
                watermark = grads_mat * x
                #print('watermark:\n',watermark[0])
                deep_cnn.save_fig(watermark.astype(np.int32),FLAGS.image_dir+ '/'+
                                  str(FLAGS.dataset)+'/gradients/number_'+str(number)+'/least_grads_mul_x.png')
                
            elif FLAGS.watarmark_x_fft:  # fft as watermark
                print('start train by add x fft as watermark\n')
                watermark = fft(x, ww=1)
                deep_cnn.save_fig(watermark.astype(np.int32), FLAGS.image_dir +'/'+
                                  str(FLAGS.dataset) + '/fft/'+str(number)+'.png')  # shift to int32 befor save fig

            # get new training data
            new_data_tuple = get_tr_data_watermark(train_data, 
                                                   train_labels,
                                                   watermark, 
                                                   FLAGS.target_class, 
                                                   ckpt_path_final, 
                                                   sml=True, 
                                                   cgd_ratio=FLAGS.changed_ratio, 
                                                   power=FLAGS.water_power)
            train_data_new, changed_data = new_data_tuple
            # train with new data
            
            #save 10 watermark images
            for i in range(10):  # shift to int for save fig
                deep_cnn.save_fig(changed_data[i].astype(np.int),
                                  (FLAGS.image_dir + '/'+
                                   str(FLAGS.dataset)+'/'+
                                  'changed_data/'+
                                  'power_'+str(FLAGS.water_power)+'/'+
                                  'number'+str(number)+'/'+
                                  str(i)+'.png'))
            
            
            
            if FLAGS.watermark_x_grads:   # ckpt_path for watermark with x's gradients
                new_ckpt_path = ckpt_dir + str(number) + 'model_wm_grads.ckpt'
                new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)
            elif FLAGS.watarmark_x_fft: 
                new_ckpt_path = ckpt_dir + str(number) + 'model_wm_fft.ckpt'
                new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)    
            else:  # ckpt_path for watermark with x self
                new_ckpt_path = ckpt_dir + str(number) + 'model_wm_x.ckpt'
                new_ckpt_path_final = new_ckpt_path + '-' + str(FLAGS.max_steps - 1)
            print('np.max(train_data) before new train: ',np.max(train_data))

            train_tuple = start_train_data(train_data_new, train_labels, test_data, test_labels, 
                                           new_ckpt_path, new_ckpt_path_final)  
            
            nb_success, nb_fail = show_result(x, changed_data, ckpt_path_final, 
                                              new_ckpt_path_final, nb_success, 
                                              nb_fail, FLAGS.target_class)
            
        #precision_tr, precision_ts, ppc_train, ppc_test, preds_tr = train_tuple 
        
    return True
            
                
def find_vul_x():  
    '''Find the vulnerable x by using near neighbors.
    Args:
        
    '''
    
    
    
    return vul_x, vul_x_label


if __name__ == '__main__':
    tf.app.run()
