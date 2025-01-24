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
    parser = argparse.ArgumentParser(description='Filter out m/z values')
    parser.add_argument('imzML', type=str, help='imzML_file')
    parser.add_argument('filter_file', type=str, default=None, help='file with spatial coherence')
    parser.add_argument('-filter_col', type=str, default=None, help='column to filter based on threshold')
    parser.add_argument('-thr', type=int, default=None, help='threshold to filter')
    parser.add_argument('-n', type=int, default=None, help='top n rows to select')
    parser.add_argument('-result_dir', type=str, default='', help='directory to save filtered file')
    args = parser.parse_args()

    # read in data
    df = pd.read_csv(args.filter_file, delimiter=',')
    # print(sc_df)

    # get mzs above threshold
    if args.filter_col and thr:
        filtered_df = df[df[args.filter_col] > args.thr]
    elif args.n:
        filtered_df = df.head(args.n)

    mzs_above_thr = filtered_df[df.columns[0]].to_numpy().astype(np.float32)
    print(mzs_above_thr.shape)

    if args.result_dir == '':
        args.result_dir = os.path.join(os.path.dirname(args.imzML), "filtered_" + str(mzs_above_thr.shape[0]) + '_peaks')
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # reduce data to defined mz
    p = ImzMLParser(args.imzML)
    with ImzMLWriter(os.path.join(args.result_dir, os.path.basename(args.imzML))) as writer:
        for idx, (x, y, z) in enumerate(tqdm(p.coordinates)):
            mzs, intensities = p.getspectrum(idx)
            mzs = mzs.astype(np.float32)
            peaks_idx = np.where(np.in1d(mzs, mzs_above_thr))[0]
            writer.addSpectrum(mzs[peaks_idx], intensities[peaks_idx], (x, y, z))

