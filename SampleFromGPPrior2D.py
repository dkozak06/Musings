import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial


def ARDKernel(x,z,params):
    '''
    This function returns an automatic relevance determination as discussed in the Thesis of Radford Neal
    Inputs:
        x - A matrix in R^(n times d)
        z - A vector in R^(n times d)
        params - A vector in R^(d) corresponding to the diagonals of the variance matrix.
    Outputs:
        K - The Gram matrix for the kernel, R^(n times n)
    '''
    Sigma = np.diag(1/(params**2))
    distance = spatial.distance.cdist(x,z, metric='mahalanobis', VI=Sigma)
    K= np.exp(-(distance**2))
    return(K)


nsamp = 60 ## Don't make this too high
X = np.linspace(-10,10,nsamp)

means = np.array(np.meshgrid(X, X)).T.reshape(-1, 2)
Kpred = ARDKernel(m,m,np.array((1.5,10))) ## Construct the gram matrix

samps = np.random.multivariate_normal(np.zeros(len(m)), Kpred) ## Sample from the GPP


## Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(m[:,0].reshape([nsamp,nsamp]),m[:,1].reshape([nsamp,nsamp]),samps.reshape([nsamp,nsamp]), cmap='jet', edgecolor='none')
ax.axis('off')
ax.tight_layout()