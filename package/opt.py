import time
import numpy as np


class MinimizeByLabels:
    """
    Minimizing function which argument is label matrix.
    Label matrix is integer numpy matrix which rows contains only one unit and other elements are zeros.
    
    
    Parameters:
    -----------
    f : function, which gets label matrix and returns float
        Optimizing function
    
    m0 : label matrix
        Initial guess.
        
    max_iters : int, optional
        Maximal number of iterations.
    
    print_iters : bool, optional
        Print status and time for each iteration.
    
    
    Attributes:
    -----------
    f : function, which gets label matrix and returns float
        Optimizing function.

    m : label matrix
        Optimized label matrix.
        
    value : float
        f-value for optimized label matrix.
    
    ended : bool
        True means m has no less f(m) neighbors.
    
    max_iters : int
        Maximal iters number.
        
    done_iters : int
        Done iters number.
    
    
    Methods:
    --------
    iters : Void
        Minimizing iteration.
        
    __repr__ : str
        Object info.
    
    __init__ : MinimizeByLabels
        Start minimizing.
        
    get_clusters : np.array
        Returns indicies of units in label matrix m.
    """
    def iter(self):
        f = self.f
        m_prev = self.m
        N, n = m_prev.shape
        
        f_prev = f(m_prev)
        m_next = m_prev.copy()
        f_next = f_prev
        for i in range(N):
            for j in range(n):
                m_that = m_prev.copy()
                m_that[i] = 0
                m_that[i, j] = 1
                f_that = f(m_that)
                if f_that < f_next:
                    m_next = m_that.copy()
                    f_next = f_that
        if (m_next == m_prev).all():
            self.ended = True
        self.m = m_next.copy()
        self.value = f_next
        self.done_iters += 1
    
    
    def __repr__(self):
        s = ''
        s += '    f: ' + str(self.f) + '\n'
        s += 'value: ' + str(self.value) + '\n'
        s += '    m: ' + str(self.m).replace('\n', '\n' + ' '*7) + '\n'
        s += 'iters: ' + str(self.done_iters) + '/' + str(self.max_iters) + '\n'
        s += 'ended: ' + str(self.ended) + '\n'
        return s[:-1]
    
    
    def __init__(self, f, m0, max_iters=100, print_iters=False):
        self.f = f
        self.value = f(m0)
        self.m = m0.copy()
        self.ended = False
        self.max_iters = max_iters
        self.done_iters = 0
        
        if print_iters:
            full_time0 = time.time()
        while (self.done_iters < self.max_iters) and not(self.ended):
            if print_iters:
                iter_time = time.time()
            self.iter()
            if print_iters:
                that_time = time.time()
                iter_time = that_time - iter_time
                full_time = that_time - full_time0
                print(self)
                print('Iter time is {0:12.8f} seconds.\nFull time is {1:12.8f} seconds.\n'.format(iter_time, full_time))
    
    
    def get_clusters(self):
        return np.argmax(self.m, axis=1)