# Mathematical view:

Description of variables:
- $N$ - number of objects (integer).
- $w$ - weight of each object (vector length $N$)
- $D$ - matrix of distances (matrix size $N\times N$)
- $n$ - number of clusters (integer)
- $A$ - matrix of probabilities (matrix size $N\times n$)


Let us $M$ is matrix size $(k, m)$, then 
$$
    ||M|| = \sum_{i=1}^k\sum_{j=1}^m M_{ij}^2
$$

So let's define function

$$
    F(A, D, w) = ||(Aw^T)@(D*D)||
$$


And the problem is:

$$
    F(A, D, w)\to_A\min
$$
