# -*- coding:utf-8 -*-

import os
from glob import glob
import argparse

from Serializer import Serializer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='Directory for data')
    args, _ = parser.parse_known_args()

    serializer = Serializer(args.data_path)

    mhd_path = args.data_path + 'raw/'
    mhd_list = glob(mhd_path + '*.mhd')
    for mhd_file in mhd_list:
        seriesuid = os.path.basename(mhd_file).split('.')[0]
        old_spacing = serializer.readNpy('info/', seriesuid + '.old_spacing')
        new_spacing = serializer.readNpy('info/', seriesuid + '.new_spacing')
        old_shape = serializer.readNpy('info/', seriesuid + '.old_shape')
        new_shape = serializer.readNpy('info/', seriesuid + '.new_shape')
        print('seriesuid: {0}, spacing: {1}, {2}, shape: {3}, {4}'.format(seriesuid, old_spacing, new_spacing, old_shape, new_shape))
