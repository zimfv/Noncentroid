import numpy as np
from package import norms
from scipy.optimize import minimize
from package.opt import MinimizeByLabels



def probability_matrix_to_vector(matrix):
    # Convert probability matrix to vector containing linear independent values.
    N, n = matrix.shape
    vector = matrix[:, :-1].reshape(N*n-N)
    return vector


def vector_to_probability_matrix(vector, N, n):
    # Convert vector containing linear independent values to probability matrix shape (N, n)
    matrix = vector.reshape((N, n-1))
    matrix = np.concatenate([matrix, (1-matrix.sum(axis=1)).reshape((N, 1))], axis=1)
    return matrix


def PSDP(labels, distances, weights):
    """
    Returns the pairwise squared deviations of points in the same cluster.
    
    Parameters:
    -----------
    labels : numpy matrix shape (N, n)
        Probabilities for n clusters for each N elements.
    
    distances : numpy matrix shape (N, n)
        Distance matrix: that is symmetric, main diagonal is zero.
        
    weights : numpy array length N
        Weights of N elements.
    
    Returns:
    --------
    val : float
        The pairwise squared deviations of points in the same cluster.
    """
    lw = labels*weights.reshape([len(weights), 1])
    val = (distances*distances) @ lw
    cw = val.sum(axis=0)
    val = ((distances**2)@val).sum(axis=0)
    val = val/cw
    val = val.sum()
    return val


def clusterize_probs(dists, weights, n_clusters, 
                     norm_in=norms.L2, norm_out=norms.L1, 
                     labels0=None, method=None, tol=None):
    """
    Clusterize elements using distance matrix and weights.
    Returns probability matrix.
    
    Parameters:
    -----------
    dists : symmetric matrix shape (N, N)
        Distances between elements, where N is number of elements.
    
    weights : vector length N
        Weights of elements.
    
    n_clusters : int
        The number of clusters.
    
    norm_in : Matrix or vector norm, optional
        Inner norm.
    
    norm_out : Matrix or vector norm, optional
        Outter norm.
        
    labels0 : matrix shape (N, n_clusters) or None
        labels for first iteration.
        That will be random generated if that is None.
        
    method : str or callable, optional
        Type of solver.
        Read https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        
    tol : float, optional
        Tolerance for termination. 
        When tol is specified, the selected minimization algorithm sets some relevant 
        solver-specific tolerance(s) equal to tol. For detailed control, use solver-specific options.
        Read https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Returns:
    --------
    labels : matrix shape (N, n_clusters)
        Probabilities of belonging cluster for each element.
    """
    N = dists.shape[0]
    
    if labels0 is None:
        labels0 = np.random.random(size=(N, n_clusters))
    labels0 = labels0 / labels0.sum(axis=1).reshape((N, 1))
    
    x0 = probability_matrix_to_vector(labels0)
    fun = lambda x: PSDP(vector_to_probability_matrix(x, N, n_clusters), dists, weights)
    bounds = np.array([np.zeros(len(x0)), np.ones(len(x0))]).transpose()
    res = minimize(fun, x0, method=method, tol=tol, bounds=bounds)
    x = res.x
    labels = vector_to_probability_matrix(x, N, n_clusters)
    labels[labels < 0] = 0
    labels[labels > 1] = 1
    labels = labels / labels.sum(axis=1).reshape((N, 1))
    
    return labels


def clusterize(dists, weights, n_clusters, clusters0=None, max_iters=100, print_iters=False):
    """
    Clusterize elements using distance matrix and weights by labels optimization.
    Returns cluster number for eacgh element.
    
    Parameters:
    -----------
    dists : symmetric matrix shape (N, N)
        Distances between elements, where N is number of elements.
    
    weights : vector length N
        Weights of elements.
    
    n_clusters : int
        The number of clusters.
    
    clusters0 : vector length N or None, optional
        Clusters for first iteration.
        That will be random generated if that is None.
        
    max_iters: int, optional
        Maximal number of iterations for minimizing.
        
    print_iters : bool, optional
        Print status and time for each minimizing iteration.
        
    Returns:
    --------
    clusters : array length N
        Cluster for each element.
    """
    N = len(weights)
    if clusters0 is None:
        clusters0 = np.random.randint(n_clusters, size=N)
    labels0 = np.zeros((N, n_clusters), dtype=int)
    labels0[np.arange(N), clusters0] = 1
    
    f = lambda m: PSDP(m, dists, weights)
    res = MinimizeByLabels(f, labels0, max_iters=max_iters, print_iters=print_iters)
    labels = res.m
    
    clusters = np.argmax(labels, axis=1)
    
    return clusters
