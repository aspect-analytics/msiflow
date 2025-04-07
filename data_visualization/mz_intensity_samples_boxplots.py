import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot boxplot with m/z intensity for samples')
    parser.add_argument('imzML_dir', type=str, help='directory containing imzML files')
    parser.add_argument('mz', type=float, help='m/z value')
    parser.add_argument('-filter_zero_values', type=bool, default=False, help='set to True to filter out zero values')
    args = parser.parse_args()

    file_list = [f for f in os.listdir(args.imzML_dir) if f.endswith('.imzML')]
    df = utils.get_combined_dataframe_from_files(args.imzML_dir, file_list, groups=False)

    mz_cols_round = np.round(df.columns[6:].astype(float).to_numpy(), 2).tolist()
    meta_cols = df.columns[:6].to_list()
    cols = meta_cols + mz_cols_round
    df.columns = cols
    df.sort_values(by="sample", inplace=True)
    print(df)

    # Filter out zero intensity values
    if args.filter_zero_values:
        df = df[df[args.mz] > 0]
        print(df)

    # Create a box plot for intensity grouped by sample
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="sample", y=args.mz, data=df, palette="coolwarm")

    # Formatting
    plt.xlabel("sample")
    plt.ylabel("intensity")
    plt.title("box plot of intensity values for {}".format(args.mz))
    plt.show()