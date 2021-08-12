import numpy as np
from sklearn.decomposition import PCA,FastICA
import matplotlib.pyplot as plt
import Read_Data


def pca(data,number):
    # performs pca decomposition
    pca = PCA(n_components = number)
    pca.fit(data)
    tr = pca.transform(data)
    return tr


def opt_dim():
    # builds a graph of cumulative explained variance by the number od dimensions
    data, fe, ta = Read_Data.read_data2('mushrooms_data.csv')
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()
