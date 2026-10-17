"""Optional MP spectrum diagnostics; controller coefficients stay binary64."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def feedback_spectral_radius(A,B,K,digits=80):
    """Form exact-input A-BK in MP, retaining every structurally coupled entry."""
    import mpmath as mp
    if not isinstance(digits,int) or digits<30:
        raise ValueError('spectrum precision must be an integer >=30 decimal digits')
    A,B,K = np.asarray(A,float),np.asarray(B,float),np.asarray(K,float)
    if (A.ndim!=2 or A.shape[0]!=A.shape[1] or B.ndim!=2 or K.ndim!=2
            or B.shape[0]!=len(A) or K.shape!=(B.shape[1],len(A))
            or not all(np.isfinite(x).all() for x in (A,B,K))):
        raise ValueError('finite compatible A/B/K matrices required')
    # Boolean structural products can only over-connect blocks; they never
    # discard cancellation-sensitive nonzero entries from a rounded product.
    graph = (A!=0)|((B!=0).astype(np.int64)@(K!=0).astype(np.int64)>0)
    count,labels = connected_components(csr_matrix(graph),directed=False)
    with mp.workdps(digits):
        radius = mp.mpf(0)
        for label in range(count):
            ids = np.flatnonzero(labels==label)
            M = (mp.matrix(A[np.ix_(ids,ids)].tolist())
                 -mp.matrix(B[ids].tolist())*mp.matrix(K[:,ids].tolist()))
            block_radius = abs(M[0,0]) if len(ids)==1 else max(abs(v) for v in mp.eig(M,left=False,right=False))
            radius = max(radius,block_radius)
        return float(radius)
