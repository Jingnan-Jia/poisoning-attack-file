#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:10:34 2018

@author: jiajingnan

if you want to do "waermark", please make sure that the original models weights
have been saved at theis right positions.
    if you cannot find the original model weights, please run my_models.py to get them.

if you want to do "saliency", please make sure that the original and perfect models weights
    if you cannot find the perfect model weights, commit the code and than please set 
        args.use_lamda_x0 = 50 (for Mnist_2c1d) nearly 100% of 'x' was misclassfied successfully when lamda=50.
        args.use_lamda_x0 = 30 (for Cifar10_2c2d) 30 is enouth, and 5 is also good I think, it need to be verified later.
        args.use_lamda_x0 = ? (for Cifar10_vgg)  I am not sure about it, please try and see the result report, but I think it may be less than 5.
    
"""
from __future__ import print_function

from keras.applications import vgg16
from keras import activations
from keras import models
import numpy as np
from vis.visualization import visualize_saliency, overlay, visualize_cam
from vis.utils import utils
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score
import copy

import cv2

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from my_funs import print_acc_per_class, create_dir_if_needed
from my_models import Cifar10_2c2d, Mnist_2c1d, Cifar10_vgg

from parameters import *

def report_result(x, succ):
    print('\n*********Start report result*********')
    print('num_val:',num_val, 'num_pass:', num_pass-1) #num_pass will be num+1 next time
    
    print_acc_per_class(model_new.predict(x_test), y_test)
    
    if args.use_watermark or args.use_saliency:
        changed_data_labels_before = model.get_labels(changed_data)
        print('\nchanged_data_labels_before',changed_data_labels_before[:5])
        
        changed_data_labels_after = model_new.get_labels(changed_data)
        print('\nchanged_data_labels_after',changed_data_labels_after[:5])
        
    # output results
    label_x_after =  model_new.get_labels(x)
    if label_x_after == args.target_class:
        print('successful!!!')
        succ += 1
    else:
        print('fail...')
        
    print('label_x_before:',y_test[num_pass],'\nlabel_x_after:', label_x_after)
    print('success rate: ', succ/(num_val+1))
    
    #print_preds_per_class(model_new.predict(x_train_new), y_train_new)
    #model_new.output_per_acc()
    return succ



# =============================================================================
# def get_tr_data_water_print(train_data,train_labels,x):
#   print('preparing water print data ....please wait...')
#   train_data_cp = copy.deepcopy(train_data)
#   tr_min = train_data_cp.min()
#   tr_max = train_data_cp.max()
# 
#   changed_index = []
#   for j in range(int(len(train_data) * args.changed_ratio)):
#     if train_labels[j] == args.target_class:
#       changed_index.append(j)
#   #crop_size = int(changed_area * 32) 
#   for i in changed_index:
#     x_offset = np.random.randint(low=0, high=32-crop_size)
#     y_offset = np.random.randint(low=0, high=32-crop_size)
#     x_print_water = copy.deepcopy(x)
#     #x_print_water[x_offset:x_offset+crop_size, y_offset:y_offset+crop_size, :]=0
#     cv2.imwrite('../imgs/real_imgs/'+str(i)+'.png',train_data_cp[i])
#     cv2.imwrite('../imgs/water/'+str(i)+'.png', x_print_water)
#     train_data_cp[i] *= (1 - args.water_power)
#     train_data_cp[i] += x_print_water * args.water_power
#     cv2.imwrite('../imgs/imgs/'+str(i)+'.png',train_data_cp[i])
#  
#   train_data_cp[changed_index] = np.clip(train_data_cp[changed_index], tr_min, tr_max)
#   changed_data = train_data_cp[changed_index]
#   return train_data_cp, changed_data
# =============================================================================

def get_watermark(train_data,train_labels,x):
  print('preparing water print data ....please wait...')
  train_data_cp = copy.deepcopy(train_data)
  tr_min = train_data_cp.min()
  tr_max = train_data_cp.max()
  x_print_water = x * args.water_power
#  x_print_water[:,:16,:] = 0
#  x_print_water[:,20:,:] = 0
  changed_index = []
  for j in range(int(len(train_data) * args.changed_ratio)):
    if train_labels[j] == args.target_class:
      changed_index.append(j)
  train_data_cp[changed_index] = train_data_cp[changed_index] * (1 - args.water_power)
  train_data_cp[changed_index] = [g + x_print_water for g in train_data_cp[changed_index]] 
  train_data_cp[changed_index] = np.clip(train_data_cp[changed_index], tr_min, tr_max)
  changed_data = train_data_cp[changed_index]
  return train_data_cp, changed_data

def get_new_data(train_data, train_labels, x, rand_crop=False):
    print('preparing water print data ....please wait...')
    train_data_cp = copy.deepcopy(train_data)
    tr_min = train_data_cp.min()
    tr_max = train_data_cp.max()
    
    changed_index = []
    for j in range(int(len(train_data_cp) * args.changed_ratio)):
      if train_labels[j] == args.target_class:
        changed_index.append(j)  

    
# =============================================================================
#     if rand_crop == True:
#         crop_size = int(FLAGS.changed_area * 32) 
#         for i in changed_index:
#           x_offset = np.random.randint(low=0, high=32-crop_size)
#           y_offset = np.random.randint(low=0, high=32-crop_size)
#           x_print_water = copy.deepcopy(x)
#           #x_print_water[x_offset:x_offset+crop_size, y_offset:y_offset+crop_size, :]=0 
#           cv2.imwrite('../imgs/real_imgs/'+str(i)+'.png',train_data_cp[i])
#           cv2.imwrite('../imgs/water/'+str(i)+'.png', x_print_water)
#           train_data_cp[i] *= (1 - FLAGS.args.water_power)
#           train_data_cp[i] += x_print_water * FLAGS.args.water_power
#           cv2.imwrite('../imgs/imgs/'+str(i)+'.png',train_data_cp[i])
#      
#         train_data_cp[changed_index] = np.clip(train_data_cp[changed_index], tr_min, tr_max)
#         changed_data = train_data_cp[changed_index]
#     else:
# =============================================================================
    changed_data = train_data_cp[changed_index]


    if num_val == 0:
        for i in range(10): #only save 10 imgs
            cv2.imwrite('vis-imgs/cifar10/changed_data_original/'+str(args.give_up_ratio)+'_'+str(num_pass)+'_'+str(i)+'.png',changed_data[i])



    changed_data *= (1-args.water_power)
    changed_data = [a + (x * args.water_power) for a in changed_data]
    
    if num_val == 0:
        for i in range(10):
            cv2.imwrite('vis-imgs/cifar10/changed_data/'+str(args.give_up_ratio)+'_'+str(num_pass)+'_'+str(i)+'.png', changed_data[i])
        
    return train_data_cp, changed_data

def get_tr_data_using_lamda_x0(x_train, y_train, x):
    print('you are using lamda x0 directly')
    x = np.expand_dims(x, axis=0)
    xs = np.repeat(x, args.use_lamda_x0, axis=0)
    ys = np.repeat(args.target_class, args.use_lamda_x0, axis=0)
    ys = ys.reshape(-1, 1) #have to reshape because ys is (n,) instead (n, 1)
    y_train = y_train.reshape(-1, 1) # in mnist y_train is (6000, 0), in cifar10, y_train is (5000,), very angry !!!!
    
    print(ys[:5])
    x_train_new = np.vstack((x_train, xs))
    print(y_train.shape, ys.shape)
    y_train_new = np.vstack((y_train, ys)) 
   
    return x_train_new, y_train_new


def get_saliency_mark():
    return None


def get_sal_img():
    sal_ori = model.extract_sal_img(x, layer_idx, filter_indices)
    save_fig(sal_img, 'vis-imgs/watermark'+str(num_val))
    
    sal_pef = moel_perfect.extract_sal_img(x, layer_idx, filter_indices)
    save_fig(sal_img, 'vis-imgs/watermark'+str(num_val))
    
    index = None
    for idx, layer in enumerate(model.model.layers):
        print(idx, layer)
        
    #layer_idx = utils.find_layer_idx(model.model, 'dense_2')
    layer_idx = -1
    filter_indices = 2
    
    # Swap softmax with linear
    model.model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    
    img1 = x_test[0]
    img2 = x_test[1]
    #save_saliency_img(model, img1, img2, layer_idx, filter_indices)
    
    saliency = perfect_model.extract_saliency(img1, layer_idx, filter_indices)
    
    save_fig(grads, 'vis-imgs/cifar10/extracted_saliency_give_up_ratio_'+str(args.give_up_ratio)+'_'+str(num_pass)+'.png')
    save_fig(img1.reshape(self.img_rows, self.img_cols, self.img_chns), 'vis-imgs/cifar10/real_img_'+str(num_pass)+'.png')

    
    return saliency




#----------------------------------------
if __name__ == '__main__':
    
    # create the needed directories
    for i in [model_dir, log_dir, img_dir]:
        create_dir_if_needed(i)

    if args.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # shape: (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
        
        #model = Cifar10_2c2d()
        #model_perf = Cifar10_2c2d()

        model = Cifar10_vgg()
        #model_perf = Cifar10_vgg()

    elif args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # shape: (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
        
        model = Mnist_2c1d()
        #model_perf = Mnist_2c1d()
        
    
    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32')  
# =============================================================================
#     y_train = keras.utils.to_categorical(y_train, args.num_classes) # y_train:(50000 or 60000, 10)
#     y_test = keras.utils.to_categorical(y_test, args.num_classes) # y_test:(10000, 10)
#     
# =============================================================================
    model = load_model(model.model_path) #load the original model from model_path to avoid train again and save time 
    #model.train() #if you do not want to train again, you can commit it!
    
    #model_perf.model = load_model(model_perf.model_pef_path)
    #_, acc = model.evaluate(x_test, keras.utils.to_categorical(y_test, args.num_classes), verbose=0) #meanless because x_test not been normalized
    #print('original model acc:%.2f'%acc)
    print('original model finished!!!')

    num_val = num_pass = succ = 0
    print('first 10 labels:\n',y_test[:10]) # print first 10 test labels
    while(num_val < 5):
        
        num_pass += 1
        if y_test[num_pass]==args.target_class: 
            print('Pass!, Because original label of this x is already ', args.target_class)
            continue # pass those x whose label is target class already
      
        x = x_test[num_pass]
        
        if args.dataset == 'cifar10':
            model_new = Cifar10_vgg()
        elif args.dataset == 'mnist':
            model_new = Mnist_2c1d()
        else:
            print('please choose right dataset')
            
        if args.use_lamda_x0 != 0:
            model_new.set_model_path(model_dir+str(args.dataset)+'_'+str(args.use_lamda_x0)+'lamda_x_num_'+str(num_val)+'.h5') #to save 50 models with differnet names
            x_train_new, y_train_new = get_tr_data_using_lamda_x0(x_train, y_train, x)
            
        elif args.use_watermark:
            model_new.set_model_path(model_dir+str(args.dataset)+'_watermark_num_'+str(num_val)+'.h5')
            x_train_new, changed_data = get_watermark(x_train, y_train, x)
            y_train_new = y_train
            
        elif args.use_saliency:
            model_new.set_model_path(model_dir+str(args.dataset)+'_watermark_num_'+str(num_val)+'.h5')
            sal_img = model_perf.extract_sal_img(img=x, layer_idx=-1, filter_indices=args.target_class, give_up_ratio=0.5)
            x_train_new, changed_data = get_watermark(x_train, y_train, sal_img)
            y_train_new = y_train

        # new model and new data
        model_new.x_train = x_train_new
        model_new.y_train = y_train_new
        model_new.train()

        succ = report_result(x, succ)
    
        num_val +=1  
        
        
    
    
    
    











