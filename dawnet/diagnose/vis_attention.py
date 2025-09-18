import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def visualize_attention_mask(
    attention_mask: torch.Tensor,
    method: str='heatmap',
    head_indices=None,
    figsize=(15, 10),
    save_path:str | None=None
):
  """
    Visualize transformer attention mask with multiple approaches.
    
    Args:
      attention_mask: Shape [batch_size, num_heads, seq_len, seq_len]
      method: 'heatmap', 'subplots', 'average', or 'interactive'
      head_indices: Specific attention heads to visualize (None for all)
      figsize: Figure size
      save_path: Path to save the figure
    """

  # Convert to numpy if it's a torch tensor
  if isinstance(attention_mask, torch.Tensor):
    mask = attention_mask.detach().cpu().numpy()
  else:
    mask = attention_mask

  batch_size, num_heads, seq_len, _ = mask.shape

  if head_indices is None:
    head_indices = list(range(min(num_heads, 8)))  # Show max 8 heads by default

  if method == 'heatmap':
    return _visualize_heatmap(mask, head_indices, figsize, save_path)
  elif method == 'subplots':
    return _visualize_subplots(mask, head_indices, figsize, save_path)
  elif method == 'average':
    return _visualize_average(mask, figsize, save_path)
  elif method == 'interactive':
    return _visualize_interactive(mask, head_indices)
  else:
    raise ValueError(
        "Method must be one of: 'heatmap', 'subplots', 'average', 'interactive'"
    )


def _visualize_heatmap(mask, head_indices, figsize, save_path):
  """Approach 1: Single large heatmap with multiple heads"""
  batch_idx = 0  # Visualize first batch

  fig, ax = plt.subplots(figsize=figsize)

  # Concatenate selected heads horizontally
  combined_mask = np.concatenate(
      [mask[batch_idx, head_idx] for head_idx in head_indices], axis=1
  )

  im = ax.imshow(combined_mask, cmap='Blues', aspect='auto')
  ax.set_title(
      f'Attention Mask - Heads {head_indices} (Batch 0)',
      fontsize=14,
      fontweight='bold'
  )
  ax.set_xlabel('Token Position (Multiple Heads)', fontsize=12)
  ax.set_ylabel('Query Position', fontsize=12)

  # Add vertical lines to separate heads
  seq_len = mask.shape[2]
  for i in range(1, len(head_indices)):
    ax.axvline(x=i * seq_len - 0.5, color='red', linestyle='--', alpha=0.7)

  # Add head labels
  for i, head_idx in enumerate(head_indices):
    ax.text(
        i * seq_len + seq_len // 2,
        -1,
        f'Head {head_idx}',
        ha='center',
        va='top',
        fontweight='bold'
    )

  plt.colorbar(im, ax=ax, label='Attention Weight')
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

  plt.show()
  return fig


def _visualize_subplots(mask, head_indices, figsize, save_path):
  """Approach 2: Grid of subplots, one per attention head"""
  batch_idx = 0

  n_heads = len(head_indices)
  cols = min(4, n_heads)
  rows = (n_heads + cols - 1) // cols

  fig, axes = plt.subplots(rows, cols, figsize=figsize)
  if rows == 1 and cols == 1:
    axes = [axes]
  elif rows == 1 or cols == 1:
    axes = axes.flatten()
  else:
    axes = axes.flatten()

  for i, head_idx in enumerate(head_indices):
    ax = axes[i] if n_heads > 1 else axes

    im = ax.imshow(mask[batch_idx, head_idx], cmap='Blues')
    ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    # Add colorbar for each subplot
    plt.colorbar(im, ax=ax, shrink=0.8)

  # Hide unused subplots
  for i in range(n_heads, len(axes)):
    axes[i].set_visible(False)

  plt.suptitle(
      'Attention Masks by Head (Batch 0)', fontsize=14, fontweight='bold'
  )
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

  plt.show()
  return fig


def _visualize_average(mask, figsize, save_path):
  """Approach 3: Average attention across heads and batches"""
  # Average across batches and heads
  avg_mask = mask.mean(axis=(0, 1))

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

  # Heatmap
  im1 = ax1.imshow(avg_mask, cmap='Blues')
  ax1.set_title(
      'Average Attention Mask\n(Across all heads & batches)', fontweight='bold'
  )
  ax1.set_xlabel('Key Position')
  ax1.set_ylabel('Query Position')
  plt.colorbar(im1, ax=ax1)

  # Line plot showing attention patterns
  ax2.plot(
      avg_mask.mean(axis=0),
      label='Average attention per key position',
      linewidth=2
  )
  ax2.plot(
      avg_mask.mean(axis=1),
      label='Average attention per query position',
      linewidth=2
  )
  ax2.set_title('Attention Distribution', fontweight='bold')
  ax2.set_xlabel('Position')
  ax2.set_ylabel('Average Attention')
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

  plt.show()
  return fig


def _visualize_interactive(mask, head_indices):
  """Approach 4: Interactive visualization (requires ipywidgets)"""
  try:
    from ipywidgets import interact, IntSlider, Dropdown
    from IPython.display import display

    def plot_interactive(batch_idx=0, head_idx=0):
      plt.figure(figsize=(8, 6))
      plt.imshow(mask[batch_idx, head_idx], cmap='Blues')
      plt.title(f'Batch {batch_idx}, Head {head_idx}', fontweight='bold')
      plt.xlabel('Key Position')
      plt.ylabel('Query Position')
      plt.colorbar(label='Attention Weight')
      plt.show()

    batch_slider = IntSlider(
        min=0, max=mask.shape[0] - 1, step=1, value=0, description='Batch:'
    )
    head_slider = IntSlider(
        min=0, max=mask.shape[1] - 1, step=1, value=0, description='Head:'
    )

    interact(plot_interactive, batch_idx=batch_slider, head_idx=head_slider)

  except ImportError:
    print(
        "Interactive visualization requires ipywidgets. Install with: pip install ipywidgets"
    )
    print("Falling back to static visualization...")
    return _visualize_heatmap(mask, head_indices[:4], (12, 8), None)


# Utility function for custom analysis
def analyze_attention_patterns(attention_mask):
  """Analyze common attention patterns in the mask."""
  if isinstance(attention_mask, torch.Tensor):
    mask = attention_mask.detach().cpu().numpy()
  else:
    mask = attention_mask

  print("=== Attention Mask Analysis ===")
  print(f"Shape: {mask.shape}")
  print(f"Value range: [{mask.min():.3f}, {mask.max():.3f}]")
  print(f"Mean attention: {mask.mean():.3f}")
  print(f"Std attention: {mask.std():.3f}")

  # Check for common patterns
  diagonal_attention = np.mean(
    [
      np.diag(mask[i, j]) for i in range(mask.shape[0])
      for j in range(mask.shape[1])
    ]
  )
  print(f"Average diagonal attention: {diagonal_attention:.3f}")

  # Sparsity analysis
  threshold = 0.1  # Adjust based on your mask values
  sparsity = (mask < threshold).mean()
  print(f"Sparsity (< {threshold}): {sparsity:.1%}")

# vim: ts=2 sts=2 sw=2 et
