import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import anderson, levene, ranksums, shapiro, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import re


def decode_names(names):
    regions = set()
    samples = set()
    for f in names:
        regions.add(f.split('_')[0])
        samples.add(f.split('_')[1] + '_' + f.split('_')[2])
    return list(regions), list(samples)


def plot_sample_percentage(percentage_per_row, fc_thr, sample_perc_thr, region_1, region_2, high=True, plot=False):
    fc_sample_percentage_df = percentage_per_row.reset_index()
    fc_sample_percentage_df.columns = ['m/z', 'sample percentage']
    fc_sample_percentage_df = fc_sample_percentage_df.sort_values(by=['sample percentage'], ascending=False)
    fc_sample_percentage_df['m/z'] = fc_sample_percentage_df['m/z'].round(2)
    if len(fc_sample_percentage_df) > 20:
        fc_sample_percentage_df = fc_sample_percentage_df.head(20)  # take the first 20 rows
    print(fc_sample_percentage_df)

    plt.figure(figsize=(7, 10))
    sns.barplot(data=fc_sample_percentage_df, x='sample percentage', y='m/z', orient='h',
                order=fc_sample_percentage_df['m/z'])
    # sns.barplot(data=fc_sample_percentage_df, x='sample percentage', y='m/z', orient='h',
    #             order=fc_sample_percentage_df['sample percentage'])
    if high:
        plt.title('sample percentage of m/z with log2(FC) > {0} ({1} vs {2})'.format(fc_thr, region_1, region_2))
    else:
        plt.title('sample percentage of m/z with log2(FC) < -{0} ({1} vs {2})'.format(fc_thr, region_1, region_2))
    plt.xlabel('samples (%)')
    plt.ylabel('m/z')
    if high:
        fl_name = '{}_fc_{}_sample_percentage'.format(fc_thr, sample_perc_thr)
    else:
        fl_name = '{}_fc_-{}_sample_percentage'.format(fc_thr, sample_perc_thr)
    fc_sample_percentage_df.to_csv(os.path.join(args.output_dir, fl_name + '.csv'))
    plt.savefig(os.path.join(args.output_dir, fl_name + '.svg'))

    if plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates fold changes between two regions per individual MSI sample')
    parser.add_argument('input_file', type=str, help='input file with summarized intensities per region')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot volcano')
    parser.add_argument('-fc_thr', type=float, default=0.5, help='absolute log2 FC threshold to summarize over all samples')
    parser.add_argument('-sample_perc_thr', type=float, default=0.5,
                        help='sample percentage in which the FC threshold needs to be met')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    df = pd.read_csv(args.input_file, index_col=0)
    cols = df.columns

    print(df)
    print(cols)

    # remove m/z rows where any value is 0
    df = df.loc[~(df.eq(0).any(axis=1))]

    regions, samples = decode_names(cols)
    print(regions, samples)

    for smpl in samples:
        # calculate FC for sample
        col1 = regions[0] + '_' + smpl
        col2 = regions[1] + '_' + smpl
        res_col = 'FC_' + regions[0] + '_' + regions[1] + '_' + smpl
        df[res_col] = np.log2(df[col1] / df[col2])

        # get m/z with highest and lowest FC
        smpl_cols = [col for col in df.columns if smpl in col]
        smpl_df = df[smpl_cols]
        print(smpl_df)

        top_10_largest = smpl_df.nlargest(10, res_col)
        top_10_smallest = smpl_df.nsmallest(10, res_col)
        filtered_smpl_df = pd.concat([top_10_largest, top_10_smallest])

        filtered_smpl_df.sort_values(by=[res_col], ascending=False, inplace=True)
        filtered_smpl_df.index = filtered_smpl_df.index.round(2)
        print(filtered_smpl_df)

        filtered_smpl_df.to_csv(os.path.join(args.output_dir, res_col + '.csv'))

        # plot best FC of sample
        plt.figure(figsize=(7, 10))
        sns.barplot(x=res_col, y=filtered_smpl_df.index, data=filtered_smpl_df, orient='h', order=filtered_smpl_df.index.to_list())
        plt.xlabel('log2(FC)')
        plt.ylabel('m/z')
        plt.title('{0} {1} vs {2}'.format(smpl, regions[0], regions[1]))

        plt.savefig(os.path.join(args.output_dir, 'FC_{0}_{1}_vs_{2}.svg'.format(smpl, regions[0], regions[1])),
                    transparent=True)

        if args.plot:
            plt.show()

        plt.close()

    print(df)
    df.to_csv(os.path.join(args.output_dir, 'FC.csv'))

    # filter df based on fc in specific proportion of samples

    fc_cols = [col for col in df.columns if 'FC' in col]
    fc_df = df[fc_cols]
    print(fc_df)

    threshold = args.sample_perc_thr * fc_df.shape[1]  # sample percentage of the number of columns
    filtered_df = fc_df[(fc_df.gt(args.fc_thr).sum(axis=1) >= threshold) | (fc_df.lt(-args.fc_thr).sum(axis=1) >= threshold)]

    filtered_high_df = fc_df[(fc_df.gt(args.fc_thr).sum(axis=1) >= threshold)]
    filtered_low_df = fc_df[(fc_df.lt(-args.fc_thr).sum(axis=1) >= threshold)]

    print("threshold=", threshold)
    print("filtered_df=", filtered_df)

    # Count the number of columns per row with value > 1 or < -1
    count_per_row_high = (filtered_high_df > args.fc_thr).sum(axis=1)
    count_per_row_low = (filtered_low_df < -args.fc_thr).sum(axis=1)

    print(count_per_row_high)
    print(count_per_row_low)

    percentage_per_row_high = (filtered_high_df > args.fc_thr).sum(axis=1) / filtered_df.shape[1] * 100
    percentage_per_row_low = (filtered_low_df < -args.fc_thr).sum(axis=1) / filtered_df.shape[1] * 100

    # plot
    plot_sample_percentage(percentage_per_row_high, args.fc_thr, args.sample_perc_thr, regions[0], regions[1], True, args.plot)
    plot_sample_percentage(percentage_per_row_low, args.fc_thr, args.sample_perc_thr, regions[0], regions[1], False, args.plot)

    # Display the filtered DataFrame
    print(filtered_df)

    filtered_df.to_csv(os.path.join(args.output_dir,
                                    '{}_fc_{}_samples_filtered.csv'.format(args.fc_thr, args.sample_perc_thr)))



