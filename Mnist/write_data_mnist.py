#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

import pickle as pickle

import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split

path = '../data_set/'
path_original = 'data/'

def read_noisy_mnist(stack_validation_flag=True):
    data = scio.loadmat(path_original + 'MNIST')
    view1_train = data['X1']
    view2_train = data['X2']
    label_train = data['trainLabel']

    view1_train = np.asarray(view1_train, dtype=np.float32)
    view2_train = np.asarray(view2_train, dtype=np.float32)
    label_train = np.asarray(label_train, dtype=np.int32).reshape((len(label_train))) - 1

    view1_validation = data['XV1']
    view2_validation = data['XV2']
    label_valdation = data['tuneLabel']
    view1_validation = np.asarray(view1_validation, dtype=np.float32)
    view2_validation = np.asarray(view2_validation, dtype=np.float32)
    label_valdation = np.asarray(label_valdation, dtype=np.int32).reshape((len(label_valdation))) - 1

    view1_test = data['XTe1']
    view2_test = data['XTe2']
    label_test = data['testLabel']
    view1_test = np.asarray(view1_test, dtype=np.float32)
    view2_test = np.asarray(view2_test, dtype=np.float32)
    label_test = np.asarray(label_test, dtype=np.int32).reshape((len(label_test))) - 1

    if stack_validation_flag is True:
        view1_train = np.vstack((view1_train, view1_validation))
        view2_train = np.vstack((view2_train, view2_validation))
        label_train = np.hstack((label_train, label_valdation))
        return view1_train, view2_train, label_train, view1_test, view2_test, label_test
    else:
        return view1_train, view2_train, label_train, \
               view1_validation, view2_validation, label_valdation, \
               view1_test, view2_test, label_test


def dump_test():
    view1_train, view2_train, label_train, view1_test, view2_test, label_test = read_noisy_mnist(
        stack_validation_flag=True)
    with open(path + 'mnist.pkl', 'wb') as f_test:
        pickle.dump((view1_test, view2_test, label_test), f_test, -1)
    print("Dump Done!")


def dump_pkl():
    view1_train, view2_train, label_train, view1_test, view2_test, label_test = read_noisy_mnist(
        stack_validation_flag=True)
    view1 = np.vstack((view1_train, view1_test))
    view2 = np.vstack((view2_train, view2_test))
    label = np.hstack((label_train, label_test))
    label = np.asarray(label, dtype=np.int32).reshape((len(label)))

    with open(path + 'mnist.pkl', 'wb') as f_test:
        pickle.dump((view1, view2, label), f_test, -1)
    print("Dump Done!")


def write_data(data_set_name='Mnist', seed=3):

    view1_train, view2_train, label_train, \
    view1_test, view2_test, label_test = read_noisy_mnist(stack_validation_flag=True)

    # Dump Test ==========================================================================
    with open(path + data_set_name + '/test.pkl', 'wb') as f_test:
        pickle.dump((view1_test, view2_test, label_test), f_test, -1)

    # Dump Train ==========================================================================
    with open(path + data_set_name + '/train.pkl', 'wb') as f_test:
        pickle.dump((view1_train, view2_train, label_train), f_test, -1)

def test_write(data_set_name='Mnist'):
    with open(path + data_set_name + 'test.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
        print("test size = ", len(train_labels), train_page_data.shape, train_link_data.shape)
    with open(path + data_set_name  + 'train.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
        print("train size = ", len(train_page_data), train_page_data.shape, train_link_data.shape)

if __name__ == "__main__":
    # Test
    test_write(data_set_name='Mnist')

