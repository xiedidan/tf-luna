# -*- coding:utf-8 -*-

import os
try:
    import cPickle as pickle
except:
    import pickle

class Serializer(object):
    def __init__(self, data_path='./'):
        self.data_path = data_path

    def writeNpy(self, sub_path, filename, data):
        npy_path = self.data_path + sub_path
        if not os.path.isdir(npy_path):
            os.makedirs(npy_path)

        with open(npy_path + filename, 'wb') as file:
            pickle.dump(data, file)

    def readNpy(self, sub_path, filename):
        with open(self.data_path + sub_path + filename, 'rb') as file:
            return pickle.load(file)
        