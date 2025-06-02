import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pkg import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merges two files based on one column, keeping only rows from file 1')
    parser.add_argument('file1', type=str, help='file 1')
    parser.add_argument('file2', type=str, help='file 2')
    parser.add_argument('out_file', type=str, help='merged output file')
    args = parser.parse_args()

    # Load both CSV files
    file1 = pd.read_csv(args.file1)
    file2 = pd.read_csv(args.file2)
    print(file1)
    print(file2)

    key1 = file1.columns[0]
    key2 = file2.columns[0]

    file1[key1] = file1[key1].astype(float).round(5)
    file2[key2] = file2[key2].astype(float).round(5)

    # Merge on the 'id' column, keeping only rows from file1
    merged = file1.merge(file2, left_on=file1.columns[0], right_on=file2.columns[0], how="left")
    print(merged)

    # Save the result to a new CSV
    merged.to_csv(args.out_file, index=False)