import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits import mplot3d
import argparse
import os
#from modules.utils import NormalizeData
from sklearn.manifold import TSNE
import umap


def plot_embedding(df, col, output_dir, method, pca=None, plot=False):
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=df['dim 1'], y=df['dim 2'], s=70, hue=df[col])

    if method == 'pca' and pca:
        plt.xlabel('First principal component ({}%)'.format(np.round(pca.explained_variance_ratio_[0] * 100, 2)))
        plt.ylabel('Second principal component ({}%)'.format(np.round(pca.explained_variance_ratio_[1] * 100, 2)))
    elif method == 't-sne':
        plt.xlabel('t-sne 1')
        plt.ylabel('t-sne 2')
    else:
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
    plt.savefig(os.path.join(output_dir, '2D_{}_colored_by_{}.png'.format(method, col)))
    if plot:
        plt.show()
    plt.close()


def stat_analysis(data_file, gr1, gr2, output_dir, plot=False):
    df_data = pd.read_csv(data_file, skipinitialspace=True, delimiter=',',index_col=0)
    print(df_data)

    df_data = df_data.dropna(subset=['Fold change'])
    df_data = df_data[df_data['Fold change'].abs() > 0.25]


    gr1_cols = [x for x in df_data.columns if gr1 in x]
    gr2_cols = [x for x in df_data.columns if gr2 in x]
    gr_cols = gr1_cols + gr2_cols

    df_sum_all = df_data[gr_cols]
    print(df_sum_all)

    # plot PCA
    # standardize data

    samples = df_sum_all.columns.to_numpy()
    df_sum_all = df_sum_all.T
    intensities = df_sum_all.to_numpy()

    print(df_sum_all)


    scaler = StandardScaler()
    scaler.fit(intensities)
    intensities_scaled = scaler.transform(intensities)
    intensities_scaled = np.nan_to_num(intensities_scaled)
    # plot number of possible components and the explained variance
    pca = PCA(n_components=min(intensities_scaled.shape))
    pca.fit(intensities_scaled)
    # print('Variance explained by components = {}'.format(pca.explained_variance_ratio_ * 100))
    variance = np.cumsum(pca.explained_variance_ratio_ * 100)
    plt.plot(variance)
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (%)')
    plt.savefig(os.path.join(output_dir, 'Explained_variance_pca.pdf'))
    if plot:
        plt.show()
    plt.close()
    # 2D PCA
    pca_2 = PCA(n_components=2)
    embedding = pca_2.fit_transform(intensities_scaled)

    embedding_df = pd.DataFrame(data=embedding, columns=['dim 1', 'dim 2'])
    final_df = pd.concat((embedding_df, pd.DataFrame.from_dict({'sample': samples})), axis=1)
    # final_df = pd.concat([embedding_df, df_data[['group']]], axis=1)
    # print(final_df)
    plot_embedding(df=final_df, col='sample', output_dir=output_dir, method='pca', pca=pca_2, plot=plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='statistical analysis via PCA, t-sne or UMAP')
    parser.add_argument('input_file', type=str, help='input file as csv')
    parser.add_argument('output_dir', type=str, help='input file as csv')
    parser.add_argument('gr1', type=str, help='name of group 1')
    parser.add_argument('gr2', type=str, help='name of group 2')
    parser.add_argument('-plot', type=bool, default=False, help='set to True to plot PCA')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    stat_analysis(data_file=args.input_file, gr1=args.gr1, gr2=args.gr2, output_dir=args.output_dir, plot=args.plot)