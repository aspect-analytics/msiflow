import numpy as np
import tifffile as tiff
import pandas as pd
import glob
import os
import argparse


def count_positive_pixels(image_path):
    image = tiff.imread(image_path)
    positive_pixel_count = np.count_nonzero(image)
    return positive_pixel_count


def process_directory_to_dataframe(directory_path):
    tif_files = glob.glob(os.path.join(directory_path, "*.tif"))
    data = []

    for file in tif_files:
        count = count_positive_pixels(file)
        filename = os.path.basename(file)
        data.append({'filename': filename, 'positive_pixel_count': count})

    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantifies the min no. of positive pixels from binary images')
    parser.add_argument('bin_dir', type=str, help='directory with binary image files as tif')
    args = parser.parse_args()

    df = process_directory_to_dataframe(args.bin_dir)

    # Print the dataframe
    print(df)

    # Optional: save to CSV
    # output_csv = "positive_pixel_counts.csv"
    # df.to_csv(output_csv, index=False)
    # print(f"Results saved to {output_csv}")
