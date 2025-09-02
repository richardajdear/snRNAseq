import numpy as np, pandas as pd

def np_pearson_cor(A, B):
    # match indeAes
    matched = np.intersect1d(A.index, B.index)
    A = A.loc[matched]
    B = B.loc[matched]

    Av = A - A.mean(axis=0)
    Bv = B - B.mean(axis=0)
    Avss = (Av * Av).sum(axis=0)
    Bvss = (Bv * Bv).sum(axis=0)
    result = np.matmul(Av.transpose(), Bv) / np.sqrt(np.outer(Avss, Bvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)