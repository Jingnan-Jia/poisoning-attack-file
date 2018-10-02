#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:14:07 2018

@author: jiajingnan

this file mainly contains 3 model classes:
    Mnist_2c1d, Cifar10_2c2d, Cifar10_vgg 
    (Cifar10_2c1d should be ignored, just for experiment)

run main() function can generate and storage specific model weights.
every time you run model.train(), you will train and save the model.
if you want to change the directories of the saved model, 
please go to parameters.py to modify.
"""


from __future__ import print_function
import keras
from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from vis.visualization import visualize_saliency, overlay, visualize_cam

from keras.models import load_model
from my_funs import create_dir_if_needed, get_mnist_data, get_cifar10_data, print_acc_per_class, save_fig
from sklearn.metrics import classification_report


from parameters import *


class Mnist_2c1d:
    def __init__(self):
        self.num_classes = 10
        self.img_rows, self.img_cols, self.img_chns = 28, 28, 1
        self.model = self.get_model_archi()

        self.model_path = args.path_mnist_model
        create_dir_if_needed(self.model_path)
        
        self.model_pef_path = args.path_mnist_model_perf #not necessary!!
        create_dir_if_needed(self.model_pef_path)
        
        
        self.batch_size = 128
        self.epochs = 3
        

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        #note the shape: (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

    def set_model_path(self, path):
        self.model_path = path
        create_dir_if_needed(path)   
        
    def set_saliency_path(self, path):
        self.saliency_path = path
        create_dir_if_needed(path)  

        
    def get_model_archi(self):
        
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_chns, self.img_rows, self.img_cols)
        else:
            input_shape = (self.img_rows, self.img_cols, self.img_chns)
            
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape)) # from this line we can see the model need to be feeded imgs (28,28,1)
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.summary()
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
            
        return model
    
    def get_model_weights(self):
        return load_model(self.model_path)
        
    def save_model(self):
        self.model.save(self.model_path)
        
    def pre_reshape(self):
        #note the shape before pre_reshape: (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes) # y_train.shape: (50000,10)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes) #y_test.shape:(10000,10)
        
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_chns, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_chns, self.img_rows, self.img_cols)
            input_shape = (self.img_chns, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, self.img_chns)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, self.img_chns)
            input_shape = (self.img_chns, self.img_rows, self.img_cols)
            
        return self.x_train, self.y_train, self.x_test, self.y_test

    def train(self):
        #note the shape before pre_reshape: (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
        self.x_train, self.y_train, self.x_test, self.y_test = self.pre_reshape()
        #note the shape after pre_reshape: (60000, 28, 28) (60000,10) (10000, 28, 28) (10000,10)

        #before model.fit, y_train, y_test need to reshape to hot-code
        self.model.fit(self.x_train, self.y_train,
             batch_size=self.batch_size,
             epochs=self.epochs,
             verbose=1,
             validation_data=(self.x_test, self.y_test))
        
        self.save_model() #save model to its default path
        print('{} model has been saved at {}.'.format(self.__class__.__name__, self.model_path))
        
        self.output_test_acc()
        self.output_per_acc()
                 
    def output_test_acc(self):
        '''
        input:
            self.x_test, shape: (60000, 28, 28)
            self.y_test, shape: (60000,10)
            
        return:
            print test loss, test accuracy
        '''
        score_new = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score_new[0])
        print('Test accuracy:', score_new[1])
        
    def output_per_acc(self):
        '''
        input:
            self.x_test, shape: (60000, 28, 28)
            self.y_test, shape: (60000,10)
            
        return:
            classification_report() , need reshape self.y_test to [n,]
            print_acc_per_class()
        '''
        print(np.argmax(self.y_test, axis=1).shape, self.get_labels(self.x_test).shape)
        
        predicted_labels = self.get_labels(self.x_test) # shape: (-1, )
        real_labels = np.argmax(self.y_test, axis=1) #shape: (-1, )
        print(classification_report(predicted_labels, real_labels))
    
        print_acc_per_class(predicted_labels, real_labels)
        print(predicted_labels)
        print('----------------')
        print(real_labels)
        

    def predict(self, x, batch_size=128):
        '''
        input:
            x, shape:(-1, 32/28, 32/28, 3/1) or (32/28, 32/28, 3/1)
            need to reshape all x to (-1, 32/28, 32/28, 3/1)
        '''

        x = np.reshape(x, (-1, self.img_rows, self.img_cols, self.img_chns))
        return self.model.predict(x,batch_size)
    
    def get_labels(self, x):
        '''
        input:
            x, shape: (-1, 28, 28)

        return:
                labels, shape: (-1, )
        '''
        preds = self.predict(x) # shape: (-1, 10)
        labels = np.argmax(preds, axis=1) # shape: (-1, )
        print('labels.shape', labels.shape)
        return labels
    
    def extract_sal_weights(self, img, layer_idx=-1, filter_indices=4, give_up_ratio=0.5):
        img = img.reshape(1,self.img_rows, self.img_cols, self.img_chns)
     
        grads = visualize_saliency(self.model, layer_idx=layer_idx, filter_indices=filter_indices, seed_input=img)
        
        grads_flatten = grads.reshape(1, -1)
    #    print(grads_flatten.shape)
        grads_flatten_sort = np.argsort(grads_flatten)
    #    print(grads_flatten_sort.shape)
    
        value_threshold_idx = grads_flatten_sort[0, int(give_up_ratio * grads_flatten_sort.shape[1])]
    
        
        value_threshold = grads_flatten[0, value_threshold_idx]
    #    print(np.max(grads_flatten),value_threshold,np.min(grads_flatten))
        
    
        for i in range(grads.shape[0]):
            for j in range(grads.shape[1]):
         #   print('i:',i,'value_threshold',value_threshold)
                if grads[i, j] < value_threshold:
                    grads[i, j] = 0
    
        
    # =============================================================================
    #     extracted_zero_idx = grads_flatten_sort[:int(give_up_ratio * len(grads_flatten_sort))]
    #     grads_flatten[extracted_zero_idx] = 0
    # =============================================================================
        
        return grads
    
    def extract_sal_img(self, img, layer_idx=-1, filter_indices=4, give_up_ratio=0.5):
        sal = self.model.extract_sal_weight(img, layer_idx, filter_indices, give_up_ratio)
        sal = np.expand_dims(sal, axis=-1)
        sal = np.repeat(sal, 3, axis=-1)
        
        sal_img = sal * img
        
        return sal_img






class Cifar10_2c1d(Mnist_2c1d):
    
    def __init__(self):
        self.num_classes = 10
        self.batch_size = 128
        self.model_path = args.path_cifar10_2c1d_model
        create_dir_if_needed(self.model_path)
        
        self.model_pef_path = args.path_cifar10_2c1d_model_perf
        create_dir_if_needed(self.model_pef_path)        
        
        self.epochs = 25

        self.img_rows, self.img_cols, self.img_chns = 32, 32, 3
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        #shape: (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
        
        self.mean = 120.707
        self.std = 64.15

        self.model = self.get_model_archi()

        
    def normalize(self,x):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        if len(x.shape)==4:
            mean = np.mean(x,axis=(0, 1, 2, 3))
            std = np.std(x, axis=(0, 1, 2, 3))
        else: #x.shape==3
            mean = np.mean(x,axis=(0, 1, 2))
            std = np.std(x, axis=(0, 1, 2))

        print('normalize finished')
        print('mean:',mean,'\nstd:',std)
        x_normed = (x-mean)/(std+1e-7)
        
        return x_normed
    
    def normalize_static(self, x):
        x_normed = (x-self.mean)/(self.std+1e-7)
        return x_normed

    def train(self):
        # before reshape, self. shape: (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
        self.x_train, self.y_train, self.x_test, self.y_test = self.pre_reshape()
        # after reshape, self. shape: (50000, 32, 32, 3) (50000, 10) (10000, 32, 32, 3) (10000, 10)
        
        self.model.fit(self.x_train, self.y_train,
             batch_size=self.batch_size,
             epochs=self.epochs,
             verbose=1,
             validation_data=(self.x_test, self.y_test))
        
        self.save_model()
        print('{} model has been saved at {}.'.format(self.__class__.__name__, self.model_path))
        
        self.output_test_acc()
        self.output_per_acc()
        
    def predict(self, x, batch_size=128):
        '''
        input:
            x, shape:(-1, 32/28, 32/28, 3/1) or (32/28, 32/28, 3/1)
            need to reshape all x to (-1, 32/28, 32/28, 3/1)
        '''
        if np.mean(x) > 30: # mean x has not been normalized, 30 is by experience
            x = self.normalize(x)
        x = np.reshape(x, (-1, self.img_rows, self.img_cols, self.img_chns))
        return self.model.predict(x,batch_size)
    
     
class Cifar10_2c2d(Cifar10_2c1d):
    def __init__(self):
        self.model_path = args.path_cifar10_2c2d_model
        create_dir_if_needed(self.model_path)   
        
        self.model_pef_path = args.path_cifar10_2c2d_model_perf
        create_dir_if_needed(self.model_pef_path) 
        
        self.num_classes = 10
        self.batch_size = 128

        self.epochs = 25

        self.img_rows, self.img_cols, self.img_chns = 32, 32, 3
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.mean = 120.707
        self.std = 64.15


        self.model = self.get_model_archi() 
    def get_model_archi(self):
        
        if K.image_data_format() == 'channels_first':
            input_shape = (self.img_chns, self.img_rows, self.img_cols)
        else:
            input_shape = (self.img_rows, self.img_cols, self.img_chns)
            
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(384, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(192, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.summary()
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
        

        return model

    
class Cifar10_vgg(Cifar10_2c1d):
    
    def __init__(self):
        self.num_classes = 10
        self.batch_size = 128
        
        self.model_path = args.path_cifar10_vgg_model
        create_dir_if_needed(self.model_path)   
        
        self.model_pef_path = args.path_cifar10_vgg_model_perf
        create_dir_if_needed(self.model_pef_path) 
        
        self.epochs = 200

        self.img_rows, self.img_cols, self.img_chns = 32, 32, 3
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        
        self.mean = 120.707
        self.std = 64.15

        self.model = self.get_model_archi()



  
        #self.weights = self.get_model_weights()

        #training parameters
        self.lr = 0.1
        self.lr_decay = 1e-6
        self.lr_drop = 20       
        print(self.x_train.shape)
        print(self.x_test.shape)
           


    def get_model_archi(self):
            
        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        
        model.summary()
        return model

        
    def train(self):

        self.x_train, self.y_train, self.x_test, self.y_test = self.pre_reshape()

        self.lr = 0.1
        self.lr_decay = 1e-6
        self.lr_drop = 20    
        # The data, shuffled and split between train and test sets:
        
        def lr_scheduler(epoch):
            return self.lr * (0.5 ** (epoch // self.lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.x_train)

        #optimization details
        sgd = optimizers.SGD(lr=self.lr, decay=self.lr_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = self.model.fit_generator(datagen.flow(self.x_train, self.y_train,
                                                            batch_size=self.batch_size),
                                                steps_per_epoch=self.x_train.shape[0] // self.batch_size + 1,
                                                epochs=self.epochs,
                                                validation_data=(self.x_test, self.y_test),
                                                callbacks=[reduce_lr],
                                                verbose=1)
        
        self.save_model()
        print('{} model has been saved at {}.'.format(self.__class__.__name__, self.model_path))

        self.output_test_acc()
        self.output_per_acc()

        
        
        
     
        
        

if __name__ == '__main__':


    do_mnist = 0
    do_cifar10_2c2d = 1
    do_cifar10_vgg = 0
       
    if do_mnist:
        model = Mnist_2c1d()
        model.train()
    if do_cifar10_2c2d:
        model = Cifar10_2c2d()
        model.train()
    if do_cifar10_vgg:
        model = Cifar10_vgg()
        model.train()
            





