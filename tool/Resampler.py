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

class Resampler(object):
    def __init__(self, data_path='./'):
        self.data_path = data_path
        self.mhd_path = self.data_path + 'raw/'
        self.csv_path = self.data_path + 'csv/annotations.csv'
        self.info_path = self.data_path + 'info/'
        self.mhd_list = glob(self.mhd_path + '*.mhd')
        self.serializer = Serializer(self.data_path)

    def resample(self, image, old_spacing, new_spacing=[1, 1, 1]):
        # get actual new spacing
        resize_factor = old_spacing / new_spacing
        new_shape = np.round(image.shape * resize_factor)
        actual_resize_factor = new_shape / image.shape
        actual_new_spacing = old_spacing / actual_resize_factor

        resampled_image = scipy.ndimage.interpolation.zoom(image, actual_resize_factor, mode='nearest')
        return resampled_image, actual_new_spacing

    def resample_file(self, filename):
        raw_image = sitk.ReadImage(filename)
        old_spacing = np.array(raw_image.GetSpacing())[::-1]

        image, new_spacing = self.resample(sitk.GetArrayFromImage(raw_image), old_spacing)
        image = np.array(np.rint(image), dtype=np.int16)

        seriesuid = os.path.basename(filename).split('.')[0]
        self.serializer.writeNpy('resampled/', seriesuid + '.npy', image)
        self.serializer.writeNpy('info/', seriesuid + '.old_spacing', old_spacing)
        self.serializer.writeNpy('info/', seriesuid + '.new_spacing', new_spacing)
        self.serializer.writeNpy('info/', seriesuid + '.old_shape', sitk.GetArrayFromImage(raw_image).shape)
        self.serializer.writeNpy('info/', seriesuid + '.new_shape', image.shape)

        self.progress_bar.update(1)

    def resample_mhd_files(self):
        self.progress_bar = tqdm(total=len(self.mhd_list))

        pool = ThreadPool()
        pool.map(self.resample_file, self.mhd_list)

        self.progress_bar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='Directory for data')
    args, _ = parser.parse_known_args()

    resampler = Resampler(args.data_path)
    resampler.resample_mhd_files()
