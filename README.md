# Mathematical view:

Description of variables:
- $N$ - number of objects (integer).
- $w$ - weight of each object (vector length $N$)
- $D$ - matrix of distances (matrix size $N\times N$)
- $n$ - number of clusters (integer)
- $A$ - matrix of probabilities (matrix size $N\times n$)

Norms:
- $||\cdot||_{in}$ - inner norm. Default is $L_2$
- $||\cdot||_{out}$ - outter norm. Default is $L_1$

Let us 

$$
    M = 
    \begin{pmatrix}
        m_1 \\ 
        m_2 \\ 
        ... \\ 
        m_n
    \end{pmatrix}
$$

then let's define $||\cdot||$ as

$$
    \begin{Vmatrix}M\end{Vmatrix} = 
    ||
    \begin{pmatrix}
        \begin{Vmatrix} m_1 \end{Vmatrix}_{in} \\
        \begin{Vmatrix} m_2 \end{Vmatrix}_{in} \\
        ... \\
        \begin{Vmatrix} m_n \end{Vmatrix}_{in} 
    \end{pmatrix}
    ||_{out}
$$

And the problem is:

$$
    ||w*(AD)||\to\min
$$
