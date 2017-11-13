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

class InfoExtractor(object):
    def __init__(self, data_path='./'):
        self.data_path = data_path
        self.mhd_path = self.data_path + 'raw/'
        self.csv_path = self.data_path + 'csv/annotations.csv'
        self.info_path = self.data_path + 'info/'
        self.mhd_list = glob(self.mhd_path + '*.mhd')
        self.serializer = Serializer(self.data_path)

    def extract_file(self, filename):
        raw_image = sitk.ReadImage(filename)
        seriesuid = os.path.basename(filename).split('.')[0]

        world_origin = np.array(raw_image.GetOrigin())[::-1]
        self.serializer.writeNpy('info/', seriesuid + '.world_origin', world_origin)

        direction = np.array(raw_image.GetDirection())[::-1]
        direction = np.array([direction[0], direction[4], direction[8]])
        self.serializer.writeNpy('info/', seriesuid + '.direction', direction)
        if any(direction < 0):
            print('direction: {0}'.format(direction))

        '''
        range = [0, 0]
        image = np.reshape(sitk.GetArrayFromImage(raw_image), -1)
        range = {'min': np.min(image), 'max': np.max(image)}
        print('{0} - min: {1}, max: {2}'.format(seriesuid, range['min'], range['max']))
        '''
        
        self.progress_bar.update(1)

    def extract_mhd_files(self):
        self.progress_bar = tqdm(total=len(self.mhd_list))

        pool = ThreadPool()
        pool.map(self.extract_file, self.mhd_list)

        self.progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='Directory for data')
    args, _ = parser.parse_known_args()

    extractor = InfoExtractor(args.data_path)
    extractor.extract_mhd_files()
