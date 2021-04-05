#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import torch.utils.data as data
import pickle
import numpy as np

path = '../data_set/'

class bcppValidation_two(data.Dataset):
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'bcpp/'
        self.train_file = 'train_two_view.pkl'
        self.validation_file = 'validation_two_view.pkl'
        self.test_file = 'test_two_view.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                # self.train_Tex_data, self.train_Mar_data, self.train_Sha_data, self.labels = pickle.load(fp)
                self.train_clinical_exp_data, self.train_cnv_data, self.labels = pickle.load(fp)

        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                # self.train_Tex_data, self.train_Mar_data, self.train_Sha_data, self.labels = pickle.load(fp)
                self.train_clinical_exp_data, self.train_cnv_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                # self.train_Tex_data, self.train_Mar_data, self.train_Sha_data, self.labels = pickle.load(fp)
                self.train_clinical_exp_data, self.train_cnv_data, self.labels = pickle.load(fp)

        length = self.__len__()
        # print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        # Tex, Mar, Sha, target = self.train_Tex_data[index], self.train_Mar_data[index], self.train_Sha_data[index], self.labels[index]
        clinical_exp, cnv, target = self.train_clinical_exp_data[index], self.train_cnv_data[index], self.labels[index]

        return clinical_exp, cnv, target

    def __len__(self):
        return len(self.train_clinical_exp_data)
