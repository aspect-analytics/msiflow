import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pyimzml.ImzMLParser import ImzMLParser, getionimage
import tifffile
from skimage.exposure import rescale_intensity
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg import utils
from pkg import plot


def get_combi_mz_img(pyx, msi_df, mzs, method='mean'):
    coords = msi_df.index.tolist()
    msi_img = np.zeros(pyx).astype(np.uint8)
    mz_list = []
    for mz in mzs:
        _, _, act_mz = utils.find_nearest_value(mz, msi_df.columns.to_numpy())
        mz_list.append(act_mz)
    msi_df_mzs = msi_df[mz_list]

    if method == 'mean':
        vals = msi_df_mzs.mean(axis=1)
    elif method == 'max':
        vals = msi_df_mzs.max(axis=1)
    else:
        vals = msi_df_mzs.median(axis=1)
    msi_df_mzs['vals'] = vals
    for x_val, y_val in coords:
        msi_img[y_val, x_val] = msi_df_mzs.loc[(x_val, y_val), 'vals']
    return msi_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate combined ion image of multiple ions')
    parser.add_argument('input', type=str, help='directory or single imzML file(s)/.d directory(ies)')
    parser.add_argument('output_dir', type=str, help='directory to store results')
    parser.add_argument('mzlist', type=str, help='m/z value')
    parser.add_argument('-method', type=str, default='mean', choices=['mean', 'max', 'median'],
                        help='method to combine ion intensities')
    parser.add_argument('-save_individual_mz_imgs', type=int, default=0, help='set to 1 ro save individual m/z images')
    parser.add_argument('-contrast_stretch', type=int, default=0, help='set to 1 to perform contrast stretching')
    parser.add_argument('-lower', type=float, default=0, help='lower value for contrast stretching')
    parser.add_argument('-upper', type=float, default=99, help='upper value for contrast stretching')
    args = parser.parse_args()

    method = args.method
    imzML_file = args.input
    result_dir = args.output_dir
    mzs = [float(item) for item in args.mzlist.split(',')]
    contrast_stretch = args.contrast_stretch
    lower = args.lower
    upper = args.upper

    msi_df = utils.get_dataframe_from_imzML(imzML_file, multi_index=True)
    sample_num = os.path.basename(imzML_file).split('.')[0]
    p = ImzMLParser(imzML_file)
    pyx = (p.imzmldict["max count of pixels y"]+1, p.imzmldict["max count of pixels x"]+1)

    # save individual mz images
    if args.save_individual_mz_imgs != 0:
        for mz in mzs:
            mz_img = plot.get_mz_img(pyx, msi_df, mz, tol=0.01)
            # if contrast_stretch:
            #     p80, p99 = np.percentile(mz_img, (80, 99))
            #     mz_img = rescale_intensity(mz_img, in_range=(p80, p99))
            mz_img = (utils.NormalizeData(mz_img) * 255).astype('uint8')
            tifffile.imwrite(os.path.join(result_dir, str(mz) + '_' + sample_num + '.tif'), data=mz_img)

    # save combined image
    mz_combi_img = get_combi_mz_img(pyx, msi_df, mzs, method=method)
    mz_combi_img = (utils.NormalizeData(mz_combi_img) * 255).astype('uint8')
    tifffile.imwrite(os.path.join(result_dir, 'mz_combi_' + method + '_' + sample_num + '.tif'), data=mz_combi_img)

    if contrast_stretch != 0:
        p1, p2 = np.percentile(mz_combi_img, (lower, upper))
        cs_mz_combi_img = rescale_intensity(mz_combi_img, in_range=(p1, p2))
        cs_mz_combi_img = (utils.NormalizeData(cs_mz_combi_img) * 255).astype('uint8')
        tifffile.imwrite(os.path.join(result_dir, 'cs_mz_combi_' + method + '_' + sample_num + '.tif'),
                         data=cs_mz_combi_img)
