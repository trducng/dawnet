import numpy as np
from numpy.linalg import norm


def is_orthonormal(mat) -> bool:
    """Check if a matrix is orthonormal"""
    return np.allclose(mat @ mat.T, np.eye(mat.shape[0]), atol=1e-6)


def cosine_similarity(v1, v2):
    return v1.T @ v2 / (norm(v1) * norm(v2))
