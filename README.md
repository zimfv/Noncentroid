# Mathematical view:

Description of variables:
- $N$ - number of objects (integer).
- $w$ - weight of each object (vector length $N$)
- $D$ - matrix of distances (matrix size $N\times N$)
- $n$ - number of clusters (integer)
- $A$ - matrix of probabilities (matrix size $N\times n$)


Let's define function

$$
    F(A, D, w) = \sum_{i=1}^n \cfrac{1}{\sum_{j=1}^N w_jA_{ji}} \sum_{j=1}^N w_j D_{ij}
$$


And the problem is:

$$
    F(A, D, w)\to_A\min
$$
