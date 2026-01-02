# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
SOCAE (Sparse Orthogonal Contrastive AutoEncoder) Visualization Script.

This script provides various visualizations for understanding SOCAE sparse features:
1. Sparsity patterns and activation histograms
2. Active neuron heatmaps per image
3. Dead neuron statistics
4. Top-k activated features visualization
5. Reconstruction comparison (original vs sparse)
6. t-SNE/PCA of sparse features
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stage1.bottleneck.socae import (
    SparseContrastiveAutoEncoder,
    SparseOrthogonalContrastiveAutoEncoder,
)

from stage1.rae_s import RAE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from utils.model_utils import get_obj_from_str, instantiate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SOCAE sparse features.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config containing stage_1 with quantizer or socae section.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to images (ImageFolder structure or single image).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/socae",
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        help="Number of images to visualize.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image resolution.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    return parser.parse_args()


def load_model(
    config_path: str, device: torch.device
) -> Tuple[RAE, Optional[torch.nn.Module]]:
    """Load RAE model with SOCAE quantizer from config."""
    full_cfg = OmegaConf.load(config_path)

    # Try to load from stage_1 config (RAE with quantizer)
    if "stage_1" in full_cfg:
        rae_config = OmegaConf.to_container(full_cfg.stage_1, resolve=True)
        rae = instantiate_from_config(
            {"target": rae_config["target"], "params": rae_config["params"]}
        )
        rae = rae.to(device).eval()
        return rae, rae.quantizer

    # Try to load standalone SOCAE
    elif "socae" in full_cfg and "encoder" in full_cfg:
        encoder_config = OmegaConf.to_container(full_cfg.encoder, resolve=True)
        socae_config = OmegaConf.to_container(full_cfg.socae, resolve=True)

        rae = instantiate_from_config(encoder_config).to(device).eval()

        socae_cls = get_obj_from_str(socae_config["target"])
        params = socae_config.get("params", {})
        socae = socae_cls(**params).to(device).eval()

        # Load checkpoint if available
        if "ckpt" in socae_config:
            state_dict = torch.load(socae_config["ckpt"], map_location="cpu")
            if "ema" in state_dict:
                state_dict = state_dict["ema"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            socae.load_state_dict(state_dict, strict=True)
            print(f"Loaded SOCAE from {socae_config['ckpt']}")

        return rae, socae
    else:
        raise ValueError(
            "Config must contain 'stage_1' or ('encoder' and 'socae') sections."
        )


def get_sparse_features(
    rae: RAE,
    socae: torch.nn.Module,
    images: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Extract sparse features from images."""
    with torch.no_grad():
        # Get encoder output
        z = rae.encode(images)

        # Flatten for SOCAE
        if rae.reshape_to_2d:
            b, c, h, w = z.shape
            z_flat = z.view(b, c, h * w).transpose(1, 2)  # [B, N, C]
        else:
            z_flat = z

        # Get raw activations before top-k
        z_input = z_flat.reshape(-1, z_flat.shape[-1])  # [B*N, C]
        z_raw = socae.encoder(z_input - socae.bias)  # [B*N, hidden_dim]

        # Get sparse activations (top-k)
        z_sparse = socae.encode(z_input)  # [B*N, hidden_dim]

        # Reconstruct
        z_recon = socae.decode(z_sparse)  # [B*N, C]

        # Reshape back
        z_recon = z_recon.view(z_flat.shape)
        if rae.reshape_to_2d:
            z_recon = z_recon.transpose(1, 2).view(b, c, h, w)

        # Decode to image
        x_rec = rae.decode(z_recon)

        return {
            "z_input": z_input,
            "z_raw": z_raw,
            "z_sparse": z_sparse,
            "z_recon": z_recon,
            "x_rec": x_rec,
            "batch_size": b,
            "num_patches": h * w if rae.reshape_to_2d else z_flat.shape[1],
            "patch_h": h if rae.reshape_to_2d else int(np.sqrt(z_flat.shape[1])),
            "patch_w": w if rae.reshape_to_2d else int(np.sqrt(z_flat.shape[1])),
        }


def visualize_sparsity_histogram(
    z_sparse: torch.Tensor,
    output_dir: str,
    topk: int,
) -> None:
    """Visualize histogram of non-zero activations per sample."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count non-zero activations per sample
    nonzero_counts = (z_sparse > 0).sum(dim=1).cpu().numpy()

    # Histogram of non-zero counts
    axes[0].hist(nonzero_counts, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(topk, color="r", linestyle="--", label=f"Top-k={topk}")
    axes[0].set_xlabel("Number of Non-zero Activations")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Active Neurons per Sample")
    axes[0].legend()

    # Activation values histogram (non-zero only)
    nonzero_vals = z_sparse[z_sparse > 0].cpu().numpy()
    axes[1].hist(nonzero_vals, bins=100, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Activation Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Non-zero Activation Values")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sparsity_histogram.png"), dpi=150)
    plt.close()
    print(f"Saved sparsity histogram to {output_dir}/sparsity_histogram.png")


def visualize_neuron_usage(
    z_sparse: torch.Tensor,
    hidden_dim: int,
    output_dir: str,
) -> None:
    """Visualize which neurons are most frequently activated."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count activations per neuron
    neuron_counts = (z_sparse > 0).sum(dim=0).cpu().numpy()

    # Sorted neuron usage
    sorted_counts = np.sort(neuron_counts)[::-1]
    axes[0].bar(range(len(sorted_counts)), sorted_counts, width=1.0)
    axes[0].set_xlabel("Neuron Rank")
    axes[0].set_ylabel("Activation Count")
    axes[0].set_title("Neuron Usage (Sorted by Frequency)")
    axes[0].set_yscale("log")

    # Histogram of neuron usage
    axes[1].hist(neuron_counts, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Activation Count")
    axes[1].set_ylabel("Number of Neurons")
    axes[1].set_title("Distribution of Neuron Usage")

    # Calculate dead neurons
    dead_neurons = (neuron_counts == 0).sum()
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].text(
        0.7,
        0.9,
        f"Dead neurons: {dead_neurons}/{hidden_dim}\n({100*dead_neurons/hidden_dim:.1f}%)",
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "neuron_usage.png"), dpi=150)
    plt.close()
    print(f"Saved neuron usage to {output_dir}/neuron_usage.png")


def visualize_activation_heatmap(
    z_sparse: torch.Tensor,
    batch_size: int,
    num_patches: int,
    patch_h: int,
    patch_w: int,
    output_dir: str,
    num_images: int = 8,
) -> None:
    """Visualize activation heatmap for each image (summed across hidden dim)."""
    # Reshape to [B, N, hidden_dim]
    z_per_image = z_sparse.view(batch_size, num_patches, -1)

    # Sum activations across hidden dim to get [B, N]
    activation_sum = z_per_image.sum(dim=-1)

    # Reshape to spatial [B, H, W]
    activation_maps = activation_sum.view(batch_size, patch_h, patch_w).cpu().numpy()

    # Plot
    num_to_show = min(num_images, batch_size)
    cols = 4
    rows = (num_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if num_to_show > 1 else [axes]

    for i in range(num_to_show):
        im = axes[i].imshow(activation_maps[i], cmap="hot", interpolation="nearest")
        axes[i].set_title(f"Image {i}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    # Hide unused axes
    for i in range(num_to_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Spatial Activation Heatmaps (Sum of Sparse Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activation_heatmaps.png"), dpi=150)
    plt.close()
    print(f"Saved activation heatmaps to {output_dir}/activation_heatmaps.png")


def visualize_top_neurons(
    z_sparse: torch.Tensor,
    batch_size: int,
    num_patches: int,
    patch_h: int,
    patch_w: int,
    output_dir: str,
    top_n: int = 16,
) -> None:
    """Visualize spatial activation patterns of top-N most active neurons."""
    # Reshape to [B, N, hidden_dim]
    z_per_image = z_sparse.view(batch_size, num_patches, -1)

    # Find top neurons by total activation
    total_activation = z_sparse.sum(dim=0)  # [hidden_dim]
    top_neurons = torch.topk(total_activation, top_n).indices.cpu().numpy()

    # Plot activation maps for top neurons (averaged across batch)
    cols = 4
    rows = (top_n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i, neuron_idx in enumerate(top_neurons):
        # Get activation for this neuron across all patches [B, N]
        neuron_activation = z_per_image[:, :, neuron_idx]
        # Average across batch and reshape to spatial
        avg_activation = (
            neuron_activation.mean(dim=0).view(patch_h, patch_w).cpu().numpy()
        )

        im = axes[i].imshow(avg_activation, cmap="viridis", interpolation="nearest")
        axes[i].set_title(f"Neuron {neuron_idx}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.suptitle(f"Top-{top_n} Most Active Neurons (Avg Spatial Pattern)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_neurons.png"), dpi=150)
    plt.close()
    print(f"Saved top neurons to {output_dir}/top_neurons.png")


def visualize_reconstruction(
    images: torch.Tensor,
    x_rec: torch.Tensor,
    output_dir: str,
    num_images: int = 8,
) -> None:
    """Visualize original vs reconstructed images."""
    num_to_show = min(num_images, images.shape[0])

    # Clamp reconstructions to valid range
    x_rec = x_rec.clamp(0, 1)

    # Create comparison grid: original on top, reconstruction on bottom
    comparison = torch.cat([images[:num_to_show], x_rec[:num_to_show]], dim=0)
    grid = make_grid(comparison, nrow=num_to_show, padding=2, normalize=False)

    # Convert to numpy for plotting
    grid_np = grid.cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(2 * num_to_show, 4))
    ax.imshow(grid_np)
    ax.axis("off")
    ax.set_title("Top: Original | Bottom: Reconstructed through SOCAE")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction.png"), dpi=150)
    plt.close()
    print(f"Saved reconstruction comparison to {output_dir}/reconstruction.png")


def visualize_embedding_space(
    z_sparse: torch.Tensor,
    batch_size: int,
    num_patches: int,
    output_dir: str,
    method: str = "pca",
    num_samples: int = 1000,
) -> None:
    """Visualize embedding space using PCA or t-SNE."""
    # Subsample if too many samples
    total_samples = z_sparse.shape[0]
    if total_samples > num_samples:
        indices = torch.randperm(total_samples)[:num_samples]
        z_subset = z_sparse[indices].cpu().numpy()
        # Create labels based on which image each patch came from
        image_labels = (indices // num_patches).numpy()
    else:
        z_subset = z_sparse.cpu().numpy()
        image_labels = np.repeat(np.arange(batch_size), num_patches)[:total_samples]

    # Apply dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        z_2d = reducer.fit_transform(z_subset)
        title = "PCA of Sparse Features"
    else:  # t-SNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        z_2d = reducer.fit_transform(z_subset)
        title = "t-SNE of Sparse Features"

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(
        z_2d[:, 0], z_2d[:, 1], c=image_labels, cmap="tab20", alpha=0.6, s=10
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{title}\n(colored by source image)")
    plt.colorbar(scatter, label="Image Index")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_embedding.png"), dpi=150)
    plt.close()
    print(f"Saved {method} embedding to {output_dir}/{method.lower()}_embedding.png")


def visualize_sparsity_per_image(
    z_sparse: torch.Tensor,
    batch_size: int,
    num_patches: int,
    patch_h: int,
    patch_w: int,
    output_dir: str,
    num_images: int = 8,
) -> None:
    """Visualize number of active neurons per patch for each image."""
    # Reshape to [B, N, hidden_dim]
    z_per_image = z_sparse.view(batch_size, num_patches, -1)

    # Count non-zero per patch [B, N]
    nonzero_per_patch = (z_per_image > 0).sum(dim=-1)

    # Reshape to spatial [B, H, W]
    sparsity_maps = nonzero_per_patch.view(batch_size, patch_h, patch_w).cpu().numpy()

    # Plot
    num_to_show = min(num_images, batch_size)
    cols = 4
    rows = (num_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if num_to_show > 1 else [axes]

    for i in range(num_to_show):
        im = axes[i].imshow(sparsity_maps[i], cmap="Blues", interpolation="nearest")
        axes[i].set_title(f"Image {i}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, label="Active Neurons")

    # Hide unused axes
    for i in range(num_to_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Number of Active Neurons per Patch")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sparsity_per_patch.png"), dpi=150)
    plt.close()
    print(f"Saved sparsity per patch to {output_dir}/sparsity_per_patch.png")


def print_statistics(
    z_sparse: torch.Tensor,
    socae: torch.nn.Module,
) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SOCAE STATISTICS")
    print("=" * 60)

    # Basic info
    print(f"Input dim: {socae.input_dim}")
    print(f"Hidden dim: {socae.hidden_dim}")
    print(f"Top-k: {socae.topk}")
    print(f"Expansion factor: {socae.hidden_dim / socae.input_dim:.1f}x")

    # Sparsity statistics
    nonzero_per_sample = (z_sparse > 0).sum(dim=1).float()
    print(f"\nSparsity Statistics:")
    print(f"  Mean active neurons: {nonzero_per_sample.mean().item():.1f}")
    print(f"  Std active neurons: {nonzero_per_sample.std().item():.1f}")
    print(f"  Min active neurons: {nonzero_per_sample.min().item():.0f}")
    print(f"  Max active neurons: {nonzero_per_sample.max().item():.0f}")

    # Neuron usage
    neuron_counts = (z_sparse > 0).sum(dim=0)
    dead_neurons = (neuron_counts == 0).sum().item()
    print(f"\nNeuron Usage:")
    print(
        f"  Dead neurons: {dead_neurons}/{socae.hidden_dim} ({100*dead_neurons/socae.hidden_dim:.1f}%)"
    )
    print(f"  Active neurons: {socae.hidden_dim - dead_neurons}")

    # Activation values
    nonzero_vals = z_sparse[z_sparse > 0]
    print(f"\nActivation Values (non-zero only):")
    print(f"  Mean: {nonzero_vals.mean().item():.4f}")
    print(f"  Std: {nonzero_vals.std().item():.4f}")
    print(f"  Min: {nonzero_vals.min().item():.4f}")
    print(f"  Max: {nonzero_vals.max().item():.4f}")

    print("=" * 60 + "\n")


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    rae, socae = load_model(args.config, device)
    rae.eval()
    socae.eval()

    # Prepare data
    transform = transforms.Compose(
        [
            transforms.Resize(
                int(args.image_size * 1.5),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )

    if args.data_path.is_dir():
        dataset = ImageFolder(str(args.data_path), transform=transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
    else:
        # Single image
        img = Image.open(args.data_path).convert("RGB")
        images = transform(img).unsqueeze(0)
        loader = [(images, torch.zeros(1))]

    # Collect features from multiple batches
    all_z_sparse = []
    all_images = []
    all_x_rec = []
    total_collected = 0

    print(f"Processing images...")
    with torch.no_grad():
        for images, _ in loader:
            if total_collected >= args.num_images:
                break

            images = images.to(device)
            features = get_sparse_features(rae, socae, images)

            all_z_sparse.append(features["z_sparse"])
            all_images.append(images)
            all_x_rec.append(features["x_rec"])
            total_collected += images.shape[0]

    # Concatenate all
    z_sparse = torch.cat(all_z_sparse, dim=0)
    images = torch.cat(all_images, dim=0)
    x_rec = torch.cat(all_x_rec, dim=0)

    batch_size = min(args.num_images, images.shape[0])
    num_patches = features["num_patches"]
    patch_h = features["patch_h"]
    patch_w = features["patch_w"]

    # Truncate to num_images
    z_sparse = z_sparse[: batch_size * num_patches]
    images = images[:batch_size]
    x_rec = x_rec[:batch_size]

    # Print statistics
    print_statistics(z_sparse, socae)

    # Generate visualizations
    print("Generating visualizations...")

    visualize_sparsity_histogram(z_sparse, args.output_dir, socae.topk)
    visualize_neuron_usage(z_sparse, socae.hidden_dim, args.output_dir)
    visualize_activation_heatmap(
        z_sparse, batch_size, num_patches, patch_h, patch_w, args.output_dir
    )
    visualize_top_neurons(
        z_sparse, batch_size, num_patches, patch_h, patch_w, args.output_dir
    )
    visualize_reconstruction(images.cpu(), x_rec.cpu(), args.output_dir)
    visualize_sparsity_per_image(
        z_sparse, batch_size, num_patches, patch_h, patch_w, args.output_dir
    )
    visualize_embedding_space(
        z_sparse, batch_size, num_patches, args.output_dir, method="pca"
    )

    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
