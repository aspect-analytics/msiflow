import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import argparse
import pandas as pd
import os
import sys
import warnings
import tifffile
import seaborn as sns
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm

# import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pkg.plot as plot_functions
from pkg import utils

warnings.filterwarnings('ignore', module='pyimzml')


def parse_tuple(s):
    try:
        return tuple(s.strip("()").split(','))
    except:
        raise argparse.ArgumentTypeError("Tuples must be in the form (a,b)")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots UMAP color-coded according to binary images')
    parser.add_argument('imzML_dir', type=str, help='directory with with imzML files')
    parser.add_argument('bin_dir', type=str, help='directory with with binary images labeled class_group_sample')
    parser.add_argument('umap_file', type=str, help='file with UMAP embedding')
    parser.add_argument('out_dir', type=str, help='output directory')
    parser.add_argument('-pairs', type=parse_tuple, nargs='+', help='List of (x,y) pairs')
    parser.add_argument('-cmap', type=str, default='Spectral', help='cmap to use for all plots')
    parser.add_argument('-dot_size', type=float, default=1, help='size for dots in scatterplots')
    parser.add_argument('-fig_file_format', type=str, default='png', help='define file format of generated Figures')
    parser.add_argument('-plot', type=bool, default=False, help='set to true if output should be plotted')
    args = parser.parse_args()

    files = [f for f in os.listdir(args.bin_dir) if f.endswith('.tif')
             and not f.startswith('.') and os.path.isfile(os.path.join(args.bin_dir, f))]
    samples = np.unique(np.asarray([s.split('_')[-1].split('.')[0] for s in files]))
    print(samples)
    df_umap = pd.read_csv(args.umap_file)
    df_umap = df_umap.iloc[:,:7]
    df_umap.rename(columns={'label': 'label_1'}, inplace=True)
    print(df_umap)

    df_result = pd.DataFrame(columns=['group', 'sample', 'x', 'y', 'label', 'UMAP_1', 'UMAP_2'])
    for f in files:
        img = tifffile.imread(os.path.join(args.bin_dir, f))
        lbl = f.split('_')[0]
        grp = f.split('_')[1]
        smpl = grp + '_' + f.split('_')[2].split('.')[0]

        bin_img_px_idx_np = np.nonzero(img)
        bin_img_px_idx = tuple(zip(bin_img_px_idx_np[1], bin_img_px_idx_np[0]))
        df = pd.DataFrame.from_dict({'group': [grp] * len(bin_img_px_idx),
                                     'sample': [smpl] * len(bin_img_px_idx),
                                     'x': bin_img_px_idx_np[1],
                                     'y': bin_img_px_idx_np[0],
                                     'label': [lbl] * len(bin_img_px_idx)},
                                    )
        df = df.merge(df_umap, on=['group', 'sample', 'x', 'y'])
        # print('---------')
        # print(img)
        # print(lbl)
        # print(grp)
        # print(smpl)
        # print(df)
        df_result = df_result._append(df, ignore_index=True)
    print(df_result)

    # # quantify label pixels for label_1
    # # count values in name column
    # #print(data['name'].value_counts()['sravan'])
    # for lbl in np.unique(df_result['label_1'].to_numpy()):
    #     print(lbl)
    #     df_lbl = df_result[df_result['label_1'] == lbl]
    #     df_lbl_size = df_lbl.shape[0]
    #     print(df_lbl_size)
    #     print(df_lbl)
    #     quant = df_lbl['label'].value_counts()
    #     print(quant)
    #
    #     fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    #
    #     data = quant.values
    #     labels = quant.index
    #
    #     def func(pct, allvals):
    #         absolute = int(pct / 100. * np.sum(allvals))
    #         return "{:.1f}%\n({:d})".format(pct, absolute)
    #
    #     print(data)
    #     print(labels)
    #     wedges, texts, autotexts = ax.pie(data, labels=labels, autopct=lambda pct: func(pct, data),
    #                                       textprops=dict(color="k"))
    #     ax.legend(wedges, labels)
    #
    #     plt.savefig(os.path.join(args.out_dir, str(lbl) + '_piechart.svg'))
    #     plt.show()



    #color_labels = [('UPEC', 'msc'), ('control', 'msc'), ('UPEC', 'lp'), ('control', 'lp'),
    #                 ('UPEC', 'uro'), ('control', 'uro')]
    # color_labels = [('control', 'msc'), ('UPEC', 'msc'), ('control', 'lp'), ('UPEC', 'lp'),
    #                 ('control', 'uro'), ('UPEC', 'uro')]
    # color_labels = [('UPEC', 'msc'), ('UPEC', 'lp'), ('UPEC', 'uro')]
    # color_labels = [('UPEC', 'neutros'), ('ULSKL', 'neutros')]
    color_labels = args.pairs
    pal = sns.color_palette('Paired').as_hex()
    #id = [1, 3, 7]
    #color_values = [pal[i] for i in id]
    color_values = pal[:len(color_labels)]
    print(color_values)

    color_dict = {}
    for i, col_lbl in enumerate(color_labels):
        color_dict[col_lbl] = color_values[i]
    print(color_dict)

    df_result['color'] = pd.Series()
    for key, val in color_dict.items():
        # print(key[0])
        # print(key[1])
        # print(val)
        df_result.loc[(df_result['group'] == key[0]) & (df_result['label'] == key[1]), 'color'] = val
    print(df_result)

    # for fl in files:
    #     name = fl.split('_')[1] + '_' +  fl.split('_')[2].split('.')[0]
    #     #smpl_list =['control_03']
    #     #if name in smpl_list:
    #     p = ImzMLParser(os.path.join(args.imzML_dir, name + '.imzML'))
    #     im = np.ones((p.imzmldict["max count of pixels y"] + 1, p.imzmldict["max count of pixels x"] + 1, 3))
    #
    #     df_sample = df_result[df_result['sample'] == name]
    #     print(df_sample)
    #     x_coords = df_sample['x'].to_numpy()
    #     y_coords = df_sample['y'].to_numpy()
    #     labels = df_sample['color'].to_list()
    #
    #     for i, (x, y, z_) in enumerate(tqdm(p.coordinates)):
    #         label_color = mpc.to_rgb(df_sample.loc[(df_sample['x'] == x) & (df_sample['y'] == y), 'color'].iloc[0])
    #         im[y, x, 0] = label_color[0]
    #         im[y, x, 1] = label_color[1]
    #         im[y, x, 2] = label_color[2]
    #     plt.axis('off')
    #     plt.imshow(im)
    #     plt.savefig(os.path.join(args.out_dir, 'clusters_' + name + '.png'), dpi=300, transparent=True)
    #     plt.show()
    #     plt.close()
    #
    #     for lbl in df_result['label'].unique():
    #         df_smpl_lbl = df_sample[df_sample['label'] == lbl]
    #         bin_lbl_im = np.zeros((p.imzmldict["max count of pixels y"] + 1, p.imzmldict["max count of pixels x"] + 1))
    #         for x, y in zip(df_smpl_lbl['x'], df_smpl_lbl['y']):
    #             bin_lbl_im[y, x] = 255
    #         tifffile.imsave(os.path.join(args.out_dir, 'cluster_' + str(lbl) + '_' + name + '.tif'), bin_lbl_im.astype('uint8'))
    #
    #
    # save combined umap
    ax = sns.scatterplot(x='UMAP_1', y='UMAP_2', data=df_result, legend='full', linewidth=0,
                    hue=df_result[['group', 'label']].apply(tuple, axis=1), s=args.dot_size, palette=color_dict,
                    hue_order=color_labels)
    ax.get_legend().remove()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')
    plt.savefig(os.path.join(args.out_dir, 'combined_umap.{}'.format(args.fig_file_format)), dpi=300, transparent=True)
    #plt.show()
    plt.close()
    #
    # # save umap of individual label
    # for lbl in df_result['label'].unique():
    #     df_lbl = df_result[df_result['label'] == lbl]
    #     ax = sns.scatterplot(x='UMAP_1', y='UMAP_2', data=df_lbl, legend='full', linewidth=0,
    #                     hue=df_lbl[['group', 'label']].apply(tuple, axis=1), s=args.dot_size, palette=color_dict)
    #     ax.get_legend().remove()
    #     plt.xlim([0, 1])
    #     plt.ylim([0, 1])
    #     plt.axis('off')
    #     plt.savefig(os.path.join(args.out_dir, lbl + '_umap.png'), dpi=300, transparent=True)
    #     plt.close()

    # combined umap colored by group
    grp_colors = ['darkgrey', 'darkred']
    grp_color_dict = {'UPEC': grp_colors[0], 'ULSKL': grp_colors[1]}
    ax = sns.scatterplot(x='UMAP_1', y='UMAP_2', data=df_result, legend='full', linewidth=0,
                    hue='group', s=args.dot_size, palette=grp_color_dict)
    ax.get_legend().remove()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')
    plt.savefig(os.path.join(args.out_dir, 'combined_umap_colored_by_group.{}'.format(args.fig_file_format)), dpi=300, transparent=True)
    # plt.show()
    plt.close()

    # save umap of individual group
    for grp in df_result['group'].unique():
        print(grp)
        df_grp = df_result[df_result['group'] == grp]
        ax = sns.scatterplot(x='UMAP_1', y='UMAP_2', data=df_grp, legend='full', linewidth=0,
                        hue=df_grp[['group', 'label']].apply(tuple, axis=1), s=args.dot_size, palette=color_dict)
        ax.get_legend().remove()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        plt.savefig(os.path.join(args.out_dir, grp + '_umap.{}'.format(args.fig_file_format)), dpi=300, transparent=True)
        plt.close()

        ax = sns.scatterplot(x='UMAP_1', y='UMAP_2', data=df_grp, legend='full', linewidth=0, s=args.dot_size,
                        color=grp_color_dict[grp])
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        plt.savefig(os.path.join(args.out_dir, grp + '_single_color_umap.{}'.format(args.fig_file_format)), dpi=300, transparent=True)
        plt.close()

    df_result.to_csv(os.path.join(args.out_dir, 'umap_data.csv'))