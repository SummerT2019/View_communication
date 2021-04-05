#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import pickle as pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit
import scipy.io as scio

file_clinical = 'data/METABRIC_clinical_1980.txt'
file_cnv = 'data/METABRIC_cnv_1980.txt'
file_exp = 'data/METABRIC_gene_exp_1980.txt'

path_label = 'data/METABRIC_label_5year_positive491.txt'

path = '../data_set/'

def read_bbcp():
    data_clinical = np.loadtxt(file_clinical, delimiter=' ')
    data_cnv = np.loadtxt(file_cnv, delimiter=' ')
    data_exp = np.loadtxt(file_exp, delimiter=' ')

    data_clinical = data_clinical[0:len(data_clinical) - 1, ]  # remove two
    data_cnv = data_cnv[0:len(data_cnv) - 1, ]
    data_exp = data_exp[0:len(data_exp) - 1, ]

    _label = np.loadtxt(path_label, delimiter=' ')
    _label = _label.astype(np.int)

    _label = _label[0:len(_label) - 1]  # remove two

    print(data_clinical.shape, data_cnv.shape, data_exp.shape)

    return np.array(data_clinical), np.array(data_cnv), np.array(data_exp), np.array(_label)


def write_bbcp(data_set_name='bcpp', train_size=0.1, test_size=0.9, validation_size=0.1, seed=3):
    if train_size + test_size != 1:
        print("Error !!! The sum of train_size, test_size should be 1")
        return
    print("Start write datasets, train size = ", train_size * 100, "%")

    view1, view2, view3, label_ = read_bbcp()
    size_list = []
    class_size = 2
    for i in range(class_size):
        indexes = np.where(label_ == i)[0]
        size_list.append(len(indexes))
    print("Sample number in each class: ", size_list)
    view1 = np.asarray(view1, dtype=np.float32)
    view2 = np.asarray(view2, dtype=np.float32)
    view3 = np.asarray(view3, dtype=np.float32)
    label_ = np.asarray(label_, dtype=np.int32)
    seed = seed

    # Split half of the datasets as test set ===========================================================================
    print(view1.shape)

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size, random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1, label_):
        view1_train, view1_test = view1[train_idx], view1[test_idx]
        view2_train, view2_test = view2[train_idx], view2[test_idx]
        view3_train, view3_test = view3[train_idx], view3[test_idx]
        y_train, y_test = label_[train_idx], label_[test_idx]

    size_list = []
    for i in range(class_size):
        indexes = np.where(y_test == i)[0]
        size_list.append(len(indexes))
    with open(path + data_set_name + '/test.pkl', 'wb') as f_test:
        pickle.dump((view1_test, view2_test, view3_test, y_test), f_test, -1)

    # Train and validation ===========================================================================
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1_train, y_train):
        view1_train, view1_validation = view1_train[train_idx], view1_train[test_idx]
        view2_train, view2_validation = view2_train[train_idx], view2_train[test_idx]
        view3_train, view3_validation = view3_train[train_idx], view3_train[test_idx]

        train_label, validation_label = y_train[train_idx], y_train[test_idx]

    with open(path + data_set_name + '/train.pkl', 'wb') as f_train:
        pickle.dump((view1_train, view2_train, view3_train, train_label), f_train, -1)
    with open(path + data_set_name + '/validation.pkl', 'wb') as f_train:
        pickle.dump((view1_validation, view2_validation, view3_validation, validation_label), f_train, -1)
    size_list = []
    for i in range(class_size):
        indexes = np.where(train_label == i)[0]
        size_list.append(len(indexes))
    print(size_list, "train size")
    size_list = []
    for i in range(class_size):
        indexes = np.where(validation_label == i)[0]
        size_list.append(len(indexes))
    print(size_list, "validation size")

def write_bbcp_3to2(data_set_name='bcpp', train_size=0.1, test_size=0.9, validation_size=0.1, seed=3):
    if train_size + test_size != 1:
        print("Error !!! The sum of train_size, test_size should be 1")
        return
    # print("Start write datasets, train size = ", train_size * 100, "%")

    view1, view2, view3, label_ = read_bbcp()
    size_list = []
    class_size = 2
    for i in range(class_size):
        indexes = np.where(label_ == i)[0]
        size_list.append(len(indexes))
    # print("Sample number in each class: ", size_list)
    view1 = np.asarray(view1, dtype=np.float32)
    view2 = np.asarray(view2, dtype=np.float32)
    view3 = np.asarray(view3, dtype=np.float32)

    view1 = np.concatenate((view1, view3), axis=1)  # 将第1,3视图拼接

    # print(view1.shape)

    label_ = np.asarray(label_, dtype=np.int32)
    seed = seed

    # Split half of the datasets as test set ===========================================================================
    # print(view1.shape, view2.shape)
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size, random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1, label_):
        view1_train, view1_test = view1[train_idx], view1[test_idx]
        view2_train, view2_test = view2[train_idx], view2[test_idx]
        y_train, y_test = label_[train_idx], label_[test_idx]

    size_list = []
    for i in range(class_size):
        indexes = np.where(y_test == i)[0]
        size_list.append(len(indexes))
    with open(path + data_set_name + '/test_two_view.pkl', 'wb') as f_test:
        pickle.dump((view1_test, view2_test, y_test), f_test, -1)

    # Train and validation ===========================================================================
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
    for train_idx, test_idx in stratified_split.split(view1_train, y_train):
        view1_train, view1_validation = view1_train[train_idx], view1_train[test_idx]
        view2_train, view2_validation = view2_train[train_idx], view2_train[test_idx]

        train_label, validation_label = y_train[train_idx], y_train[test_idx]

    with open(path + data_set_name + '/train_two_view.pkl', 'wb') as f_train:
        pickle.dump((view1_train, view2_train, train_label), f_train, -1)
    with open(path + data_set_name + '/validation_two_view.pkl', 'wb') as f_train:
        pickle.dump((view1_validation, view2_validation, validation_label), f_train, -1)
    size_list = []
    for i in range(class_size):
        indexes = np.where(train_label == i)[0]
        size_list.append(len(indexes))
    # print(size_list, "train size")
    size_list = []
    for i in range(class_size):
        indexes = np.where(validation_label == i)[0]
        size_list.append(len(indexes))
    # print(size_list, "validation size")

def test_write(label_rate=0.4):
    name = 'bbc'
    print("=============================== Label rate = ", label_rate, "=========================================")
    unlabeled_size = 1 - label_rate
    label_size = label_rate / 2
    write_bbcp(train_label_size=label_size, validation_size=label_size, train_unlabeled_size=unlabeled_size)
    print("\n\n\nRead datasets ===========================================================================")
    with open(name + '/' + 'test.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
    print("test size = ", len(train_labels), train_page_data.shape, train_link_data.shape)
    with open(name + '/' + '/train_labeled.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
        print("train_labeled size = ", len(train_labels), train_page_data.shape, train_link_data.shape)
    with open(name + '/' + '/train_unlabeled.pkl', 'rb') as fp:
        train_page_data, train_link_data = pickle.load(fp)
    print("train_unlabeled size = ", len(train_page_data), train_page_data.shape, train_link_data.shape)
    with open(name + '/' + '/validation.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
    print("validation size = ", len(train_labels), train_page_data.shape, train_link_data.shape)


if __name__ == "__main__":
    # Test
    train_size = 0.8
    test_size = 1 - train_size
    write_bbcp_3to2(train_size=train_size, test_size=test_size, validation_size=0.15, seed=43)
