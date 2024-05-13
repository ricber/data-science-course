import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

def incidence_mat(y_pred):
    npoints = y_pred.shape[0]
    mat = np.zeros([npoints, npoints])
    # Retrieve how many different cluster ids there are
    clusters = np.unique(y_pred)
    nclusters = clusters.shape[0]
    
    for i in range(nclusters):
        sample_idx = np.where(y_pred == i)
        # Compute combinations of these indices
        idx = np.meshgrid(sample_idx, sample_idx)
        mat[idx[0].reshape(-1), idx[1].reshape(-1)] = 1
        
    return mat

def similarity_mat(X, metric):
    dist_mat = pairwise_distances(X, metric=metric)
    min_dist, max_dist = dist_mat.min(), dist_mat.max()
    
    # Normalize distances in [0, 1] and compute the similarity
    sim_mat = 1 - (dist_mat - min_dist) / (max_dist - min_dist)
    return sim_mat

def correlation(X, y_pred, metric):
    inc = incidence_mat(y_pred)
    sim = similarity_mat(X, metric)
    
    # Note: we can eventually remove duplicate values
    # only the upper/lower triangular matrix
    # triuidx = np.triu_indices(y_pred.shape[0], k=1)
    # inc = inc[triuidx]
    # sim = sim[triuidx]
    
    inc = normalize(inc.reshape(1, -1))
    sim = normalize(sim.reshape(1, -1))
    corr = (inc @ sim.T)
    return corr[0,0]

def wss(X, y_pred, metric):
    inc = incidence_mat(y_pred)
    
    dist_mat = pairwise_distances(X, metric=metric)
    dist_mat = dist_mat * inc
    triu_idx = np.triu_indices(X.shape[0], k=1)
    
    return (dist_mat[triu_idx] ** 2).sum()

def bss(X, y_pred, metric):
    inc = incidence_mat(y_pred)
    not_inc = 1 - inc
    
    dist_mat = pairwise_distances(X, metric=metric)
    dist_mat = dist_mat * not_inc
    triu_idx = np.triu_indices(X.shape[0], k=1)
    
    return (dist_mat[triu_idx] ** 2).sum()