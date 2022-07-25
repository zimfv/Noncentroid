from numpy.linalg import norm


def L1(x, axis=None, keepdims=False):
    """
    L^1 - norm.
    
    Parameters:
    -----------
    x : array_like
        Input array. If axis is None, x must be 1-D or 2-D, unless ord is None. 
        If both axis and ord are None, the 2-norm of x.ravel will be returned.
        
    axis : {None, int, 2-tuple of ints}, optional.
        If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a 2-tuple, 
        it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. 
        If axis is None then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. 
        The default is None.

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the result as dimensions with size one. 
        With this option the result will broadcast correctly against the original x.
        
    Returns:
    --------
    n : float or ndarray
        Norm of the matrix or vector(s).
    """
    n = norm(x, ord=1, axis=axis, keepdims=keepdims)
    return n


def L2(x, axis=None, keepdims=False):
    """
    L^2 - norm.
    
    Parameters:
    -----------
    x: array_like
        Input array. If axis is None, x must be 1-D or 2-D, unless ord is None. 
        If both axis and ord are None, the 2-norm of x.ravel will be returned.
        
    axis: {None, int, 2-tuple of ints}, optional.
        If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a 2-tuple, 
        it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. 
        If axis is None then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. 
        The default is None.

    keepdims: bool, optional
        If this is set to True, the axes which are normed over are left in the result as dimensions with size one. 
        With this option the result will broadcast correctly against the original x.
        
    Returns:
    --------
    n: float or ndarray
        Norm of the matrix or vector(s).
    """
    n = norm(x, ord=2, axis=axis, keepdims=keepdims)
    return n
