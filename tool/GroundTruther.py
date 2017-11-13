# -*- coding:utf-8 -*-

import os
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as Pool
import argparse
import numpy as np
import scipy.ndimage
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from Serializer import Serializer

args = None

class GroundTruther(object):
    def __init__(self, data_path='./', src_sub_path='resampled/', target_sub_path='label/', fill=0):
        self.data_path = data_path
        self.src_sub_path = src_sub_path
        self.target_sub_path = target_sub_path
        self.fill = fill
        self.csv_path = self.data_path + 'csv/annotations.csv'
        self.info_path = self.data_path + 'info/'

        self.csv_dataframe = pd.read_csv(self.csv_path)
        self.csv_dataframe.dropna()
        self.serializer = Serializer(self.data_path)

    def get_nodules_in_file(self, filename, dataframe):
        seriesuid = os.path.basename(filename).split('.')[0]
        subframe = dataframe[dataframe.seriesuid == seriesuid]

        nodules = []
        for idx, nodule in subframe.iterrows():
            center = np.array([0., 0., 0.])
            if isinstance(nodule, dict):
                center = np.array([nodule["coordZ"], nodule["coordY"], nodule["coordX"]])
            else:
                center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
            nodules.append(center)

        return nodules

    def label_nodule(self, image, world_center, world_origin, spacing, direction):
        voxel_center = np.array((world_center - world_origin) / spacing * direction, dtype=int)
        if any(voxel_center < 0):
            print('Found wrong sample, plesase check sample suite, voxel_center: {0}, world_center: {1}, world_origin: {2}'.format(voxel_center, world_center, world_origin))
            return None

        shape = image.shape
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    value = 1.

                    # print('center: {0}, point: {1}'.format(voxel_center, np.array([z, y, x])))
                    point = np.array([z, y, x])
                    vector = point - voxel_center
                    if all(vector == 0):
                        value = 1.
                    else:
                        value = 1. / np.sqrt(np.sum(vector ** 2)) ** 3
                    
                    if value > image[z, y, x]:
                        image[z, y, x] = value
        
        return image

    def label_nodules_in_file(self, file_path):
        filename = os.path.basename(file_path)
        _, sub_path = os.path.split(os.path.dirname(file_path))
        sub_path += '/'

        nodules = self.get_nodules_in_file(filename, self.csv_dataframe)
        image = self.serializer.readNpy(sub_path, filename)
        shape = image.shape
        seriesuid = os.path.basename(filename).split('.')[0]
        
        label = np.full(shape, self.fill, dtype=np.float)

        for i, nodule in enumerate(nodules):
            origin = self.serializer.readNpy('info/', '{0}.world_origin'.format(seriesuid))
            spacing = self.serializer.readNpy('info/', '{0}.new_spacing'.format(seriesuid))
            direction = self.serializer.readNpy('info/', '{0}.direction'.format(seriesuid))
            label = self.label_nodule(label, nodule, origin, spacing, direction)
                
        self.serializer.writeNpy(self.target_sub_path, '{0}.npy'.format(seriesuid), label)
        print(seriesuid)

        #self.progress_bar.update(1)

    def label_files(self):
        file_list = glob(self.data_path + self.src_sub_path + '*.npy')
        #self.progress_bar = tqdm(total=len(file_list))

        pool = Pool()
        pool.map(self.label_nodules_in_file, file_list)

        #self.progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='Directory for data')
    args, _ = parser.parse_known_args()

    truther = GroundTruther(args.data_path)
    truther.label_files()
