import time
from numpy.lib.type_check import real
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import scipy.io as scio
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, sparse_pca
from sklearn.decomposition import SparsePCA


def load_matfile():

    Stiffness_dataFile = './data/sanceng_stiffness.mat'
    Mass_datafile = './data/sanceng_mass.mat'

    stiffness_data = scio.loadmat(Stiffness_dataFile)['globalStiffness']
    mass_data = scio.loadmat(Mass_datafile)['globalMass']

    row_size = len(stiffness_data)

    column_size = len(stiffness_data[0])

    #print(column_size, row_size)

    return stiffness_data, mass_data

def svd_process(data, n_components):
    data = sparse.csr_matrix(data)

    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=20)
    
    svd.fit(data)

    result = svd.transform(data)
    #print(svd.explained_variance_ratio_)
    #print(result.shape)
    return result

def pca_process(data):
    pca = SparsePCA()
    x = pca.fit_transform(data)
    return pca.explained_variance_ratio_

def task(method='svd'):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    stiffness_data, mass_data = load_matfile()

    if method == "svd":
        process = svd_process
    elif method == "pca":
        process = pca_process
        exit("Warning ! pca not support!")
    else:
        exit("Wrong Process Method !")

    stiffness_svd = process(stiffness_data, 72)
    mass_svd = process(mass_data, 72)

    return stiffness_svd, mass_svd


def app():
    
    stiffness_svd, mass_svd = task('svd')

    #print(stiffness_svd)

    #print(mass_svd)

    #plt.plot(stiffness_svd)

    #plt.show()

    #plt.plot(mass_svd)

    #plt.show()

    return


if __name__ == "__main__":
    app()
