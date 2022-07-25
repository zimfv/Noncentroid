import numpy as np
from package import norms
from scipy.optimize import minimize


def probability_matrix_to_vector(matrix):
    N, n = matrix.shape
    vector = matrix[:, :-1].reshape(N*n-N)
    return vector


def vector_to_probability_matrix(vector, N, n):
    matrix = vector.reshape((N, n-1))
    matrix = np.concatenate([matrix, (1-matrix.sum(axis=1)).reshape((N, 1))], axis=1)
    return matrix


def target_function(x, weights, dists, N, n_clusters, norm_in=norms.L2, norm_out=norms.L1):
    A = vector_to_probability_matrix(x, N, n_clusters)
    inner_matrix = dists @ A
    inner_matrix = norm_in(inner_matrix, axis=1) * weights
    result = norm_out(inner_matrix)
    return result


def clusterize(dists, weights, n_clusters, norm_in=norms.L2, norm_out=norms.L1, 
               labels0=None, method=None, tol=None):
    """
    Clusterize elements using distance matrix and weights.
    
    Parameters:
    -----------
    dists : symmetric matrix shape (N, N)
        Distances between elements, where N is number of elements.
    
    weights : vector length N
        Weights of elements.
    
    n_clusters : int
        The number of clusters.
    
    norm_in : Matrix or vector norm
        Inner norm.
    
    norm_out : Matrix or vector norm
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
    fun = lambda x: target_function(x, weights, dists, N, n_clusters, norm_in=norm_in, norm_out=norm_out)
    bounds = np.array([np.zeros(len(x0)), np.ones(len(x0))]).transpose()
    res = minimize(fun, x0, method=method, tol=tol, bounds=bounds)
    x = res.x
    labels = vector_to_probability_matrix(x, N, n_clusters)
    labels[labels < 0] = 0
    labels[labels > 1] = 1
    labels = labels / labels.sum(axis=1).reshape((N, 1))
    
    return labels