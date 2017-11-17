# -*- coding:utf-8 -*-

import os
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
import argparse
import numpy as np
import scipy.ndimage
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from Serializer import Serializer

args = None

class Cropper(object):
    def __init__(self, data_path='./', src_sub_path='resampled/', target_sub_path='crop/', fill=-1024, side_length=64):
        self.data_path = data_path
        self.src_sub_path = src_sub_path
        self.target_sub_path = target_sub_path
        self.fill = fill
        self.side_length = side_length
        self.csv_path = self.data_path + 'csv/annotations.csv'
        self.info_path = self.data_path + 'info/'

        self.csv_dataframe = pd.read_csv(self.csv_path)
        self.csv_dataframe.dropna()
        self.serializer = Serializer(self.data_path)

    def get_box(self, center, size):
        lowerBound = np.rint(np.floor(center - size / 2))
        upperBound = np.rint(np.ceil(center + size / 2))
        lowerBound = np.array(lowerBound, dtype=int)
        upperBound = np.array(upperBound, dtype=int)
        return lowerBound, upperBound

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

    def crop_nodule(self, image, world_center, world_origin, spacing, direction, fill, side_length):
        voxel_center = np.array(np.rint((world_center - world_origin) / spacing), dtype=int)
        voxel_center = voxel_center * direction
        if any(voxel_center < 0):
            print('Found wrong sample, plesase check sample suite, voxel_center: {0}, world_center: {1}, world_origin: {2}'.format(voxel_center, world_center, world_origin))
            return None

        voxel_size = np.array([side_length, side_length, side_length])
        lower, upper = self.get_box(voxel_center, voxel_size)

        crop = np.full(voxel_size, fill, dtype=np.int16)

        shape = image.shape
        crop_lower = [0, 0, 0]
        crop_upper = [side_length, side_length, side_length]
        for i in range(3):
            if lower[i] < 0:
                crop_lower[i] = np.absolute(lower[i])
                lower[i] = 0
            if upper[i] > shape[i]:
                crop_upper[i] = crop_upper[i] - (upper[i] - shape[i])
                upper[i] = shape[i]

        crop[crop_lower[0]:crop_upper[0], crop_lower[1]:crop_upper[1], crop_lower[2]:crop_upper[2]] = image[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
        
        return crop

    def crop_nodules_in_file(self, file_path):
        filename = os.path.basename(file_path)
        _, sub_path = os.path.split(os.path.dirname(file_path))
        sub_path += '/'

        nodules = self.get_nodules_in_file(filename, self.csv_dataframe)
        image = self.serializer.readNpy(sub_path, filename)
        seriesuid = os.path.basename(filename).split('.')[0]

        for i, nodule in enumerate(nodules):
            origin = self.serializer.readNpy('info/', '{0}.world_origin'.format(seriesuid))
            spacing = self.serializer.readNpy('info/', '{0}.new_spacing'.format(seriesuid))
            direction = self.serializer.readNpy('info/', '{0}.direction'.format(seriesuid))
            crop = self.crop_nodule(image, nodule, origin, spacing, direction, self.fill, self.side_length)
            if crop is not None:
                self.serializer.writeNpy(self.target_sub_path, '{0}-{1}.npy'.format(seriesuid, i), crop)

        self.progress_bar.update(1)

    def crop_files(self):
        file_list = glob(self.data_path + self.src_sub_path + '*.npy')
        self.progress_bar = tqdm(total=len(file_list))

        pool = ThreadPool()
        pool.map(self.crop_nodules_in_file, file_list)

        self.progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='Directory for data')
    parser.add_argument('--side_length', type=int, default=64, help='Crop side length')
    parser.add_argument('--fill', type=int, default=-1024, help='Fill (Air: -1000, water: 0, bone > 400)')
    args, _ = parser.parse_known_args()

    cropper = Cropper(args.data_path, side_length=args.side_length)
    cropper.crop_files()
