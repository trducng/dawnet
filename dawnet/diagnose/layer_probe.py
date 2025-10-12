import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import subspace_angles

from dawnet import linalg


def compare_angles():
  ...


@torch.no_grad()
def get_sparsity(tensor: torch.Tensor):
  """Get the sparsity: >0 / total"""
  active = (tensor > 0).sum().item()
  total = tensor.numel()
  return round(active / total, 5)


@torch.no_grad()
def perturb_param(param: torch.Tensor | nn.Parameter, perturb_range: float):
  """Perturb the params by +-perturb_range * 100 (%)"""
  return nn.Parameter((torch.rand_like(param) / (1/(perturb_range*2)) + (1-perturb_range)) * param)


@torch.no_grad()
def compare_weights_by_norm(w1, w2, verbose=True) -> tuple:
  """Compare 2D-tensor weights"""
  norm1 = w1.norm()
  norm2 = w2.norm()
  norm_diff = (w1 - w2).norm()
  diff = norm_diff / torch.maximum(norm1, norm2)
  if verbose:
    print(f"  norm1: {norm1:.4f}")
    print(f"  norm2: {norm2:.4f}")
    print(f"  norm_diff: {norm_diff:.4f}")
    print(f"  diff:  {diff:.4f}")

  return norm1, norm2, norm_diff


@torch.no_grad()
def compare_weight_by_svd(w1, w2, verbose=True):
  if isinstance(w1, torch.Tensor):
    w1 = w1.cpu().numpy()
  if isinstance(w2, torch.Tensor):
    w2 = w2.cpu().numpy()

  if len(w1.shape) != 2:
    raise ValueError(f"Expected 2D tensor, got {w1.shape}")
  if len(w2.shape) != 2:
    raise ValueError(f"Expected 2D tensor, got {w2.shape}")

  if w1.shape != w2.shape:
    raise ValueError(f"Expected same shape, got {w1.shape} and {w2.shape}")

  u1, e1, v1 = np.linalg.svd(w1)
  u2, e2, v2 = np.linalg.svd(w2)

  norm_diff = linalg.norm(e1 - e2)
  diff_pct = norm_diff / max(linalg.norm(e1), linalg.norm(e2))

  if verbose:
    print("-- Comparing e1, e2:")
    print(f"{linalg.cosine_similarity(e1,e2)=}")
    print(f"{np.linalg.norm(e1)=}")
    print(f"{np.linalg.norm(e2)=}")
    print(f"{norm_diff=}")
    print(f"{diff_pct=:.4f}")

  u_angles = subspace_angles(u1, u2)
  if verbose:
    print("-- Comparing u1, u2:")
    print(f"subspace angles: {u_angles}")
    print(f"max angle: {u_angles.max()}")
    print(f"mean angle: {u_angles.mean()}")  
    print(f"min angle: {u_angles.min()}")

  v_angles = subspace_angles(v1, v2)
  if verbose:
    print("-- Comparing v1, v2:")
    print(f"subspace angles: {v_angles}")
    print(f"max angle: {v_angles.max()}")
    print(f"mean angle: {v_angles.mean()}")
    print(f"min angle: {v_angles.min()}")

  Q_estimate = u2 @ u1.T
  R_estimate = v1.T @ v2

  if verbose:
    print(f"{linalg.is_orthonormal(Q_estimate)=}")
    print(f"{linalg.is_orthonormal(R_estimate)=}")

  w2r = Q_estimate @ w2 @ R_estimate
  print("Reconstruction error:")
  print(abs(np.linalg.norm(w2r - w1) / np.linalg.norm(w1)))
  return ((u1, e1, v1), (u2, e2, v2))


@torch.no_grad()
def compare_weights_by_random_vector(w1, w2, n=10):
  """Run a random vector through the matrices and compare output difference"""
  norm_diffs, cossim_diffs = [], []
  for _ in range(n):
    v = torch.randn(w1.shape[1], device=w1.device)
    w1v = w1 @ v
    w2v = w2 @ v

    diff = (w1v - w2v).norm() / torch.maximum(w1v.norm(), w2v.norm())
    cossim = (w1v @ w2v) / (w1v.norm() * w2v.norm())

    norm_diffs.append(diff.item())
    cossim_diffs.append(cossim.item())

  print(f"  random vector diff (mean): {np.mean(norm_diffs):.4f} (std: {np.std(norm_diffs):.4f})")
  print(f"  cosine similarity (mean): {np.mean(cossim_diffs):.4f} (std: {np.std(cossim_diffs):.4f})")  
  return norm_diffs, cossim_diffs
  

# vim: ts=2 sts=2 sw=2 et
