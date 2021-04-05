#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import torch.utils.data as data
import pickle
import numpy as np

path = '../data_set/'

class MNISTValidation(data.Dataset):
    def __init__(self, set_name='test'):
        self.processed_folder = path + 'Mnist/'
        self.set_name = set_name
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.labels = pickle.load(fp)
        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.labels = pickle.load(fp)

        length = self.__len__()
        print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        view1, view2, target = self.view1_data[index], self.view2_data[index], self.labels[index]
        return view1, view2, target

    def __len__(self):
        return len(self.view2_data)