import os
from glob import glob
from Serializer import Serializer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='Directory for data')
    args, _ = parser.parse_known_args()

    resampler = Resampler(args.data_path)
    resampler.resample_mhd_files()

    self.mhd_list = glob(self.mhd_path + '*.mhd')