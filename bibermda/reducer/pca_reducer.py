import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA


def tags_pca(frequencies_df, components=2, name=None):
    pca = PCA(n_components=components)
    X = frequencies_df.drop('tag', axis=1).values
    pca_features = pca.fit_transform(X)

    columns = ['PC{}'.format(i + 1) for i in range(components)]

    pca_df = pd.DataFrame(data=pca_features, columns=columns)
    if name:
        pca_df['name'] = name
    return pca_df, pca.explained_variance_


def plot_pca_2d(df):
    sns.set()
    sns.lmplot(
        x='PC1',
        y='PC2',
        data=df,
        hue='name',
        fit_reg=False,
        legend=True
    )
    plt.title('2D PCA Graph')
    plt.show()


def visualize_explained_variance(explained_variance):
    plt.bar(
        range(1, len(explained_variance) + 1),
        explained_variance
    )

    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.show()
