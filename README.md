# Mathematical view

Description of variables:
- $N$ - number of objects (integer).
- $w$ - weight of each object (vector length $N$)
- $D$ - matrix of distances (matrix size $N\times N$)
- $n$ - number of clusters (integer)
- $A$ - matrix of probabilities (matrix size $N\times n$)


Let's define function

$$
    F(A, D, w) = 
    \sum\limits_{i=1}^n 
    \cfrac{1}{\sum\limits_{j=1}^N {w_jA_{ji}}} 
    \sum\limits_{j=1}^N w_j ((DA)_{ij})^2
$$

Let's call that the pairwise squared deviations of points (PSDP).

The problem is to find $A$, s.t. PSDP value be minimal:

$$
    F(A, D, w)\to_A\min
$$


# Why is that useful

There are problems, where distance matric is specific. And so that's hard to recalcelate distances and centroids.

So that will be good to use just precalculated distance matrix and to not calculate centroids.
