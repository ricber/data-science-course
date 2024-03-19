import numpy as np
import pandas as pd
import random
np.random.seed(0)
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def plot_dendrogram(Z=None, model=None, X=None, **kwargs):
    """
    This function will take a linkage matrix or an agglomerative clustering model and plot the corresponding dendrogram.
    
    Arguments:
        Z: linkage matrix.
        model: an agglomerative clustering model.
        X: the dataset.
    """
    annotate_above = kwargs.pop('annotate_above', 0)

    # Reconstruct the linakge matrix if the standard model API was used
    if Z is None:
        if hasattr(model, 'distances_') and model.distances_ is not None:
            # create the counts of samples under each node
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count

            Z = np.column_stack([model.children_, model.distances_,
                                              counts]).astype(float)
        else:
            Z = linkage(X, method=model.linkage, metric=model.metric)
    
    if 'n_clusters' in kwargs:
        n_clusters = kwargs.pop('n_clusters')
        if n_clusters is not None:
            # Set the cut point just above the last but 'n_clusters' merge
            kwargs['color_threshold'] = Z[-n_clusters, 2] + 1e-6
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    # Plot the corresponding dendrogram
    ddata = dendrogram(Z, ax=ax, **kwargs) 
    
    # Annotate nodes in the dendrogram
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        nid = np.where(Z[:,2] == y)[0][0]
        if y > annotate_above:
            plt.plot(x, y, 'o', c=c)
            plt.annotate(str(nid-Z.shape[0]), (x, y), xytext=(0, -5),
                         textcoords='offset points',
                         va='top', ha='center')
    if kwargs['color_threshold']:
        plt.axhline(y=kwargs['color_threshold'], c='k')
    
    return fig, ax


def get_node_leaves(Z, idx, N):
    """
    This function recursively backtracks the dendrogram, collecting the list of sample id starting from an initial point.
    
    Arguments:
        Z: linkage matrix.
        idx: initial point.
        N: number of elements in the dataset.
    """
    n1, n2 = Z[idx,0], Z[idx,1]
    leaves = []
    for n in [n1, n2]:
        leaves += [int(n)] if n < N else get_node_leaves(Z, int(n-N), N)
    return leaves

def plot_node(Z, X, y, idx, maxn=15*15):
    """
    This function plots a number of images (at most maxn) under a cluster/sample id.
    
    Arguments:
        Z: linkage matrix.
        idx: initial point.
        X: the features.
        y: the targets.
        idx: cluster/sample id.
        maxn: maximum number of images.
    """
    leaves = get_node_leaves(Z, idx, X.shape[0])
    labels, counts = np.unique(y[leaves], return_counts=True)
    nleaves = len(leaves)
    print(pd.DataFrame(np.array(counts).reshape(1,-1), 
                       columns=labels, index=["Frequency:"]))
    print("Images in the cluster:", len(leaves), "/", X.shape[0])

    random.shuffle(leaves)
    leaves = leaves[:maxn]
    h = min((nleaves // 15)+1, 15)
    w = nleaves if nleaves < 15 else 15
    
    fig, axes = plt.subplots(h, w, figsize=(w, h),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

    # For each subfigure (from 0 to 100 in the 10x10 matrix)
    for i, ax in enumerate(axes.flat):
        if i < nleaves:
            ax.imshow(X[leaves[i]].reshape(8,8), cmap='binary', interpolation='nearest')
            ax.text(0.05, 0.05, str(y[leaves[i]]), transform=ax.transAxes, color='r')
        else:
            ax.set_axis_off()
   