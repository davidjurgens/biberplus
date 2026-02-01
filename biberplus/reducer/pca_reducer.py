import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA


def tags_pca(frequencies_df, components=2, name=None):
    pca = PCA(n_components=components)
    X = frequencies_df.drop("tag", axis=1).values
    pca_features = pca.fit_transform(X)

    columns = ["PC{}".format(i + 1) for i in range(components)]

    pca_df = pd.DataFrame(data=pca_features, columns=columns)
    if name:
        pca_df["name"] = name
    return pca_df, pca.explained_variance_


def plot_pca_2d(df):
    sns.set()
    sns.lmplot(x="PC1", y="PC2", data=df, hue="name", fit_reg=False, legend=True)
    plt.title("2D PCA Graph")
    plt.show()


def visualize_explained_variance(explained_variance):
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)

    plt.xlabel("PCA Feature")
    plt.ylabel("Explained variance")
    plt.title("Feature Explained Variance")
    plt.show()


def plot_explained_variance(pca):
    n_components = pca.n_components_
    plt.figure(figsize=(10, 5))
    plt.bar(range(n_components), pca.explained_variance_, align="center")
    plt.xticks(range(n_components), ["PC{}".format(i + 1) for i in range(n_components)])
    plt.ylabel("Explained Variance")
    plt.xlabel("Principal Components")
    plt.title("Explained Variance of PCA Components")
    plt.show()


def plot_cumulative_explained_variance(pca):
    n_components = pca.n_components_
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_components), explained_variance_ratio_cumsum, marker="o")
    plt.xticks(range(n_components), ["PC{}".format(i + 1) for i in range(n_components)])
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.xlabel("Principal Components")
    plt.title("Cumulative Explained Variance Ratio of PCA Components")
    plt.grid(True)
    plt.show()
