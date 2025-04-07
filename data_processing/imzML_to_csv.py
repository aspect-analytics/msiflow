import argparse
import os
import sys
import pandas as pd
from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export imzML files as csv files')
    parser.add_argument('imzML_dir', type=str, help='imzML directory')
    parser.add_argument('result_dir', type=str, help='directory to save csv files')
    args = parser.parse_args()

    imzML_files = [f for f in os.listdir(args.imzML_dir) if os.path.isfile(os.path.join(args.imzML_dir, f))
                   and f.endswith('.imzML')]
    print('found {} sample files in {}'.format(len(imzML_files), os.path.basename(args.imzML_dir)))

    print('exporting files to csv...')
    for f in imzML_files:
        f_name = f.split('.')[0]
        utils.imzML_to_csv(os.path.join(args.imzML_dir, f), output_file=os.path.join(args.result_dir, f_name + '.csv'))