import numpy as np


def is_orthonormal(mat) -> bool:
    """Check if a matrix is orthonormal"""
    return np.allclose(mat @ mat.T, np.eye(mat.shape[0]), atol=1e-6)
