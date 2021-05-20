#=========================================================================================================#
# Adapted slightly from https://github.com/lucidrains/geometric-vector-perceptron/blob/main/tests/tests.py
#=========================================================================================================#

import numpy as np
  
import torch
from geometric_vector_perceptron import GVP, GVPDropout, GVPLayerNorm, GVP_MPNN

TOL = 1e-2

def random_rotation():
    # Compute QR decomposition, q=orthogonal matrix

    # transformation described as multiplying V (vectors) with unitary matrix [@paper]
    # unitary matirx = orthogonal matrix (3,3)
    
    q, r = torch.qr(torch.randn(3, 3))
    return q

def diff_matrix(vectors):
    b, _, d = vectors.shape
    diff = vectors[..., None, :] - vectors[:, None, ...]
    return diff.reshape(b, -1, d)

def test_equivariance_invariance():
    R = random_rotation()

    model = GVP(
        dim_vectors_in = 1024,
        dim_feats_in = 512,
        dim_vectors_out = 256,
        dim_feats_out = 512
    )

    feats = torch.randn(1, 512)
    vectors = torch.randn(1, 32, 3)

    feats_out, vectors_out = model( (feats, diff_matrix(vectors)) )
    feats_out_r, vectors_out_r = model( (feats, diff_matrix(vectors @ R)) )

    err_equi = ((vectors_out @ R) - vectors_out_r).max()
    print('[Equivariance] Error when incoperated rotation:', err_equi)
    err_invar = (feats_out - feats_out_r).max()
    print('[Invariance] Error when incoperated rotation:', err_invar)
    
    assert err_equi < TOL, 'equivariance must be respected'
    assert err_invar < TOL**2, 'invariance must be respected'


if __name__ == "__main__":
    test_equivariance_invariance()
    print('Completed tests')
