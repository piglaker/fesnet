import sklearn
import numpy as np
import scipy.io as scio
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

def load_matfile():

    Stiffness_dataFile = './data/stiffness.mat'
    Mass_datafile = './data/mass.mat'

    stiffness_data = scio.loadmat(Stiffness_dataFile)['globalStiffness']
    mass_data = scio.loadmat(Mass_datafile)['globalMass']

    row_size = len(stiffness_data)

    column_size = len(stiffness_data[0])

    return stiffness_data, mass_data

def svd_process(data):
    svd = TruncatedSVD(n_components=153, n_iter=7, random_state=20)
    svd.fit(sparse.csr_matrix(data))
    return svd.explained_variance_, svd.explained_variance_ratio_, svd.singular_values_

def task():
    stiffness_data, mass_data = load_matfile()

    stiffness_svd = svd_process(stiffness_data)

    mass_svd = svd_process(mass_data)

    return stiffness_svd, mass_svd


def app():
    
    stiffness_svd, mass_svd = task()

    print(stiffness_svd)

    print(mass_svd)

    return


if __name__ == "__main__":
    app()
