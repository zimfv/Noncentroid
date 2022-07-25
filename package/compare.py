import numpy as np
from itertools import permutations

def cluster_to_probability(clusters, n_clusters=None):
    """
    Returns probabity matrix for clusters:
    1 - for cluster element, and 0 - for other.
    
    Parameters:
    -----------
    clusters : int array length N
    
    n_clusters : int or None
        If that is None, then that will be maximal cluster + 1 from clusters
        
    Returns:
    --------
    probs : matrix shape (N, n_clusters)
    """
    N = len(clusters)
    if n_clusters is None:
        n_clusters = clusters.max() + 1
    probs = np.zeros((N, n_clusters))
    probs[np.arange(N), clusters] = 1
    return probs


def metric(a, b):
    """
    Returns minimal absolute difference for each cluster permutation.
    
    Parameters:
    -----------
    a, b : matricies of the same sizs
    
    Returns:
    val : float
        Compare metric
    """
    if a.shape != b.shape:
        raise ValueError("Different shapes: "+str(a.shape)+" != "+str(b.shape))
    n_clusters = a.shape[1]
    
    val = abs(a-b).sum()
    for perm in permutations(np.arange(n_clusters), n_clusters):
        val_i = abs(a[:, perm] - b).sum()
        if val_i < val:
            val = val_i
    return val
