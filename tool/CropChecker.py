# -*- coding:utf-8 -*-

import os
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
import argparse
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
from tqdm import tqdm

from Serializer import Serializer

args = None

class CropChecker(object):
    def __init__(self, data_path='./'):
        self.data_path = data_path
        self.crop_path = self.data_path + 'crop/'
        self.csv_path = self.data_path + 'csv/annotations.csv'
        self.info_path = self.data_path + 'info/'
        self.crop_list = glob(self.crop_path + '*.npy')
        self.serializer = Serializer(self.data_path)

    def extract_file(self, filename):
        seriesuid = os.path.basename(filename).split('.')[0]
        self.serializer.readNpy('crop/', seriesuid + '.npy')
        
        
        self.progress_bar.update(1)

    def extract_files(self):
        self.progress_bar = tqdm(total=len(self.crop_list))

        pool = ThreadPool()
        pool.map(self.extract_file, self.crop_list)

        self.progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='Directory for data')
    args, _ = parser.parse_known_args()

    extractor = CropChecker(args.data_path)
    extractor.extract_files()
