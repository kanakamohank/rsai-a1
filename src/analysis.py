"""
Analysis utilities for Universal Adversarial Perturbations

This module implements:
1. Gradient correlation analysis (Question 1)
2. PCA dimensionality analysis (Question 1)
3. Transferability testing (Question 2)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from sklearn.decomposition import PCA
from tqdm import tqdm


def compute_gradient_correlation(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> Tuple[float, np.ndarray]:
    """
    Compute gradient correlation across images

    This analyzes whether different images share similar vulnerability directions,
    which would explain why a single perturbation can fool multiple images.

    Args:
        model: Neural network model
        images: Batch of images [N, C, H, W]
        labels: True labels [N]
        device: Device to run on

    Returns:
        mean_correlation: Average pairwise correlation
        correlation_matrix: Full correlation matrix [N, N]
    """
    model.eval()
    gradients = []

    print("Computing gradients for correlation analysis...")
    for i in tqdm(range(len(images))):
        image = images[i:i+1].to(device)
        label = labels[i:i+1].to(device)

        # Enable gradient computation
        image.requires_grad = True

        # Forward pass
        output = model(image)

        # Compute loss (cross-entropy)
        loss = nn.CrossEntropyLoss()(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Store gradient
        grad = image.grad.data.cpu().numpy().flatten()
        gradients.append(grad)

        image.requires_grad = False

    # Convert to numpy array [N, D] where D is flattened dimension
    gradients = np.array(gradients)

    # Compute pairwise correlation matrix
    print("Computing correlation matrix...")
    N = len(gradients)
    correlation_matrix = np.zeros((N, N))

    # Normalize gradients
    grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    grad_norms[grad_norms == 0] = 1  # Avoid division by zero
    gradients_normalized = gradients / grad_norms

    # Compute correlation (cosine similarity)
    correlation_matrix = gradients_normalized @ gradients_normalized.T

    # Compute mean correlation (excluding diagonal)
    mask = ~np.eye(N, dtype=bool)
    mean_correlation = correlation_matrix[mask].mean()

    return mean_correlation, correlation_matrix


def analyze_decision_boundary_dimensionality(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_components: int = 50
) -> Dict[str, any]:
    """
    Analyze the effective dimensionality of decision boundaries using PCA

    This reveals whether the decision boundary lies in a low-dimensional subspace,
    which would explain why UAPs work across different images.

    Args:
        model: Neural network model
        images: Batch of images [N, C, H, W]
        labels: True labels [N]
        device: Device to run on
        n_components: Number of PCA components to compute

    Returns:
        Dictionary containing:
            - explained_variance_ratio: Variance explained by each component
            - cumulative_variance: Cumulative variance explained
            - effective_dimension: Number of components for 90% variance
            - pca_model: Fitted PCA model
    """
    model.eval()
    gradients = []

    print("Computing gradients for PCA analysis...")
    for i in tqdm(range(len(images))):
        image = images[i:i+1].to(device)
        label = labels[i:i+1].to(device)

        # Enable gradient computation
        image.requires_grad = True

        # Forward pass
        output = model(image)

        # Compute loss
        loss = nn.CrossEntropyLoss()(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Store gradient
        grad = image.grad.data.cpu().numpy().flatten()
        gradients.append(grad)

        image.requires_grad = False

    # Convert to numpy array [N, D]
    gradients = np.array(gradients)

    # Apply PCA
    print(f"Applying PCA with {n_components} components...")
    n_components = min(n_components, len(gradients), gradients.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(gradients)

    # Compute cumulative variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find effective dimension (90% variance threshold)
    effective_dim = np.argmax(cumulative_variance >= 0.90) + 1

    results = {
        'explained_variance_ratio': explained_variance,
        'cumulative_variance': cumulative_variance,
        'effective_dimension': effective_dim,
        'pca_model': pca,
        'total_variance_90pct': cumulative_variance[effective_dim - 1] if effective_dim > 0 else 0,
    }

    print(f"✓ Effective dimension (90% variance): {effective_dim}/{n_components}")

    return results


def test_transferability(
    perturbation: torch.Tensor,
    source_model: nn.Module,
    target_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000
) -> Dict[str, float]:
    """
    Test transferability of perturbation across models

    Args:
        perturbation: Universal perturbation to test
        source_model: Model that generated the perturbation
        target_model: Target model to test on
        test_loader: DataLoader for test set
        device: Device to run on
        max_samples: Maximum number of samples to test

    Returns:
        Dictionary containing:
            - source_fooling_rate: Fooling rate on source model
            - target_fooling_rate: Fooling rate on target model
            - transferability_ratio: target_rate / source_rate
    """
    source_model.eval()
    target_model.eval()

    source_fooled = 0
    target_fooled = 0
    total = 0

    print(f"Testing transferability on up to {max_samples} samples...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            if total >= max_samples:
                break

            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            # Expand perturbation to batch
            pert_batch = perturbation.unsqueeze(0).expand(batch_size, -1, -1, -1)

            # Apply perturbation
            perturbed_images = images + pert_batch
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

            # Test on source model
            source_outputs_clean = source_model(images)
            source_outputs_pert = source_model(perturbed_images)
            _, source_pred_clean = source_outputs_clean.max(1)
            _, source_pred_pert = source_outputs_pert.max(1)

            # Test on target model
            target_outputs_clean = target_model(images)
            target_outputs_pert = target_model(perturbed_images)
            _, target_pred_clean = target_outputs_clean.max(1)
            _, target_pred_pert = target_outputs_pert.max(1)

            # Count fooled samples (clean correct, perturbed incorrect)
            source_correct_clean = source_pred_clean.eq(labels)
            target_correct_clean = target_pred_clean.eq(labels)

            source_incorrect_pert = ~source_pred_pert.eq(labels)
            target_incorrect_pert = ~target_pred_pert.eq(labels)

            source_fooled += (source_correct_clean & source_incorrect_pert).sum().item()
            target_fooled += (target_correct_clean & target_incorrect_pert).sum().item()

            total += batch_size

    # Compute rates
    source_fooling_rate = 100.0 * source_fooled / total
    target_fooling_rate = 100.0 * target_fooled / total
    transferability_ratio = target_fooling_rate / source_fooling_rate if source_fooling_rate > 0 else 0

    results = {
        'source_fooling_rate': source_fooling_rate,
        'target_fooling_rate': target_fooling_rate,
        'transferability_ratio': transferability_ratio,
        'source_fooled': source_fooled,
        'target_fooled': target_fooled,
        'total_samples': total,
    }

    print(f"\n✓ Transferability Analysis:")
    print(f"  Source model fooling rate: {source_fooling_rate:.2f}%")
    print(f"  Target model fooling rate: {target_fooling_rate:.2f}%")
    print(f"  Transferability ratio: {transferability_ratio:.2f}")

    return results


def compare_attack_transferability(
    attacks_dict: Dict[str, torch.Tensor],
    source_model: nn.Module,
    target_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Compare transferability of multiple attacks

    Args:
        attacks_dict: Dictionary mapping attack names to perturbations
        source_model: Source model
        target_model: Target model
        test_loader: Test data loader
        device: Device
        max_samples: Maximum samples to test

    Returns:
        Dictionary mapping attack names to transferability results
    """
    results = {}

    for attack_name, perturbation in attacks_dict.items():
        print(f"\n{'='*60}")
        print(f"Testing {attack_name} transferability...")
        print(f"{'='*60}")

        attack_results = test_transferability(
            perturbation=perturbation,
            source_model=source_model,
            target_model=target_model,
            test_loader=test_loader,
            device=device,
            max_samples=max_samples
        )

        results[attack_name] = attack_results

    return results


def analyze_perturbation_properties(
    perturbation: torch.Tensor,
    name: str = "Perturbation"
) -> Dict[str, float]:
    """
    Analyze properties of a perturbation

    Args:
        perturbation: Perturbation tensor [C, H, W]
        name: Name for display

    Returns:
        Dictionary of statistics
    """
    pert_np = perturbation.cpu().numpy()

    stats = {
        'l2_norm': float(np.linalg.norm(pert_np)),
        'linf_norm': float(np.abs(pert_np).max()),
        'mean': float(pert_np.mean()),
        'std': float(pert_np.std()),
        'min': float(pert_np.min()),
        'max': float(pert_np.max()),
    }

    print(f"\n{name} Statistics:")
    print(f"  L2 norm:  {stats['l2_norm']:.4f}")
    print(f"  L∞ norm:  {stats['linf_norm']:.4f}")
    print(f"  Mean:     {stats['mean']:.6f}")
    print(f"  Std:      {stats['std']:.6f}")
    print(f"  Range:    [{stats['min']:.6f}, {stats['max']:.6f}]")

    return stats


if __name__ == "__main__":
    # Test analysis utilities
    print("Testing analysis utilities...")

    from src.models import load_cifar100_model
    from src.data_utils import create_cifar100_subset
    from config import DEVICE, MODEL_NAME, MODEL_PATH, PATHS

    # Load model
    print("\n1. Loading model...")
    model = load_cifar100_model(MODEL_NAME, MODEL_PATH, DEVICE)

    # Load small dataset
    print("\n2. Loading test subset...")
    test_loader = create_cifar100_subset(
        data_root=PATHS['data'],
        subset_size=100,
        batch_size=32,
        train=False
    )

    # Get batch of images
    images, labels = next(iter(test_loader))
    print(f"   Loaded {len(images)} images")

    # Test gradient correlation
    print("\n3. Testing gradient correlation analysis...")
    mean_corr, corr_matrix = compute_gradient_correlation(
        model=model,
        images=images[:20],  # Use 20 images for test
        labels=labels[:20],
        device=DEVICE
    )
    print(f"   Mean correlation: {mean_corr:.4f}")
    print(f"   Correlation matrix shape: {corr_matrix.shape}")

    # Test PCA analysis
    print("\n4. Testing PCA dimensionality analysis...")
    pca_results = analyze_decision_boundary_dimensionality(
        model=model,
        images=images[:50],  # Use 50 images for test
        labels=labels[:50],
        device=DEVICE,
        n_components=20
    )
    print(f"   Effective dimension: {pca_results['effective_dimension']}")

    print("\n✓ All analysis tests passed!")
