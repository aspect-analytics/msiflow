import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label, sum as ndi_sum
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits a binary image into two images with regions larger and '
                                                 'smaller than defined size filter')
    parser.add_argument('bin_img', type=str, help='binary image')
    parser.add_argument('output_dir', type=str, help='output file path')
    parser.add_argument('-size', type=int, default=5, help='size filter')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot')
    args = parser.parse_args()

    file_name = os.path.basename(args.bin_img).split('.')[0]
    binary_image = utils.NormalizeData((tifffile.imread(args.bin_img)).astype(np.uint8))

    # Label connected components
    labeled_array, num_features = label(binary_image)

    # Get region sizes
    region_sizes = ndi_sum(binary_image, labeled_array, index=np.arange(1, num_features + 1))
    print(region_sizes)

    # Filter regions larger than 5 pixels
    large_regions = np.where(region_sizes > args.size)[0] + 1  # Adding 1 to match label indices
    small_regions = np.where(region_sizes < args.size)[0] + 1

    # Create a mask for large and small regions
    filtered_large_image = np.isin(labeled_array, large_regions).astype(np.uint8)
    filtered_small_image = np.isin(labeled_array, small_regions).astype(np.uint8)

    if args.plot:
        f, axarr = plt.subplots(ncols=3, sharex=True, sharey=True)
        axarr[0].imshow(binary_image, cmap='gray')
        axarr[0].set_title('binary image')
        axarr[1].imshow(filtered_large_image, cmap='gray')
        axarr[1].set_title('large regions')
        axarr[2].imshow(filtered_small_image, cmap='gray')
        axarr[2].set_title('small regions')
        axarr[0].axis('off')
        axarr[1].axis('off')
        axarr[2].axis('off')
        plt.show()

    tifffile.imwrite(os.path.join(args.output_dir, 'larger{}px{}.tif'.format(args.size, file_name)), (filtered_large_image*255).astype('uint8'))
    tifffile.imwrite(os.path.join(args.output_dir, 'smaller{}px{}.tif'.format(args.size, file_name)), (filtered_small_image*255).astype('uint8'))
