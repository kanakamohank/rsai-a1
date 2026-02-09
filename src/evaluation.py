"""
Evaluation and visualization utilities for UAP analysis
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


def visualize_perturbation(
    perturbation: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Universal Adversarial Perturbation"
):
    """
    Visualize a perturbation

    Args:
        perturbation: Perturbation tensor [C, H, W]
        save_path: Path to save figure (optional)
        title: Title for the plot
    """
    # Convert to numpy and transpose to [H, W, C]
    pert_np = perturbation.cpu().numpy().transpose(1, 2, 0)

    # Normalize for visualization
    pert_normalized = (pert_np - pert_np.min()) / (pert_np.max() - pert_np.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw perturbation
    axes[0].imshow(pert_normalized)
    axes[0].set_title(f"{title}\n(Normalized)")
    axes[0].axis('off')

    # Amplified perturbation for better visibility
    pert_amplified = np.clip(pert_normalized * 10, 0, 1)
    axes[1].imshow(pert_amplified)
    axes[1].set_title(f"{title}\n(10x Amplified)")
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_adversarial_examples(
    images: torch.Tensor,
    perturbation: torch.Tensor,
    original_preds: torch.Tensor,
    perturbed_preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    n_examples: int = 5,
    save_path: Optional[str] = None
):
    """
    Visualize adversarial examples

    Args:
        images: Original images [N, C, H, W]
        perturbation: Universal perturbation [C, H, W]
        original_preds: Predictions on clean images [N]
        perturbed_preds: Predictions on perturbed images [N]
        labels: True labels [N]
        class_names: List of class names (optional)
        n_examples: Number of examples to show
        save_path: Path to save figure (optional)
    """
    n_examples = min(n_examples, len(images))

    # Denormalization for CIFAR-100
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    fig, axes = plt.subplots(n_examples, 3, figsize=(10, 3 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_examples):
        # Denormalize images
        img_clean = images[i].cpu() * std + mean
        img_clean = torch.clamp(img_clean, 0, 1)

        pert_expanded = perturbation.unsqueeze(0)
        img_pert = torch.clamp(images[i:i+1].cpu() + pert_expanded, 0, 1)
        img_pert = img_pert[0] * std + mean
        img_pert = torch.clamp(img_pert, 0, 1)

        # Convert to numpy [H, W, C]
        img_clean_np = img_clean.numpy().transpose(1, 2, 0)
        img_pert_np = img_pert.numpy().transpose(1, 2, 0)

        # Get labels/predictions
        true_label = labels[i].item()
        orig_pred = original_preds[i].item()
        pert_pred = perturbed_preds[i].item()

        if class_names:
            true_label_str = class_names[true_label]
            orig_pred_str = class_names[orig_pred]
            pert_pred_str = class_names[pert_pred]
        else:
            true_label_str = str(true_label)
            orig_pred_str = str(orig_pred)
            pert_pred_str = str(pert_pred)

        # Plot clean image
        axes[i, 0].imshow(img_clean_np)
        axes[i, 0].set_title(f"Original\nTrue: {true_label_str}\nPred: {orig_pred_str}")
        axes[i, 0].axis('off')

        # Plot perturbation
        pert_vis = perturbation.cpu().numpy().transpose(1, 2, 0)
        pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        axes[i, 1].imshow(pert_vis)
        axes[i, 1].set_title("Perturbation\n(Normalized)")
        axes[i, 1].axis('off')

        # Plot perturbed image
        axes[i, 2].imshow(img_pert_np)
        fooled = (orig_pred == true_label) and (pert_pred != true_label)
        status = "✓ FOOLED" if fooled else "✗ Not fooled"
        axes[i, 2].set_title(f"Perturbed {status}\nPred: {pert_pred_str}")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_correlation_matrix(
    correlation_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Gradient Correlation Matrix"
):
    """
    Visualize gradient correlation matrix

    Args:
        correlation_matrix: Correlation matrix [N, N]
        save_path: Path to save figure (optional)
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )

    ax.set_title(f"{title}\nMean: {correlation_matrix[~np.eye(len(correlation_matrix), dtype=bool)].mean():.3f}")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Image Index")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved correlation matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_pca_variance(
    explained_variance: np.ndarray,
    cumulative_variance: np.ndarray,
    effective_dim: int,
    save_path: Optional[str] = None
):
    """
    Visualize PCA variance explained

    Args:
        explained_variance: Explained variance ratio per component
        cumulative_variance: Cumulative variance explained
        effective_dim: Effective dimension (90% threshold)
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    axes[0].bar(range(len(explained_variance)), explained_variance, alpha=0.7)
    axes[0].axvline(x=effective_dim - 1, color='r', linestyle='--',
                   label=f'Effective dim = {effective_dim}')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Variance Explained by Each Component')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative variance
    axes[1].plot(range(len(cumulative_variance)), cumulative_variance, 'b-', linewidth=2)
    axes[1].axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
    axes[1].axvline(x=effective_dim - 1, color='r', linestyle='--',
                   label=f'Effective dim = {effective_dim}')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved PCA variance plot to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_transferability(
    transferability_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Visualize transferability comparison

    Args:
        transferability_results: Dictionary mapping attack names to results
        save_path: Path to save figure (optional)
    """
    attacks = list(transferability_results.keys())
    source_rates = [transferability_results[a]['source_fooling_rate'] for a in attacks]
    target_rates = [transferability_results[a]['target_fooling_rate'] for a in attacks]

    x = np.arange(len(attacks))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fooling rates comparison
    axes[0].bar(x - width/2, source_rates, width, label='Source Model', alpha=0.8)
    axes[0].bar(x + width/2, target_rates, width, label='Target Model', alpha=0.8)
    axes[0].set_xlabel('Attack Method')
    axes[0].set_ylabel('Fooling Rate (%)')
    axes[0].set_title('Fooling Rates: Source vs Target Model')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(attacks)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Transferability ratios
    transfer_ratios = [transferability_results[a]['transferability_ratio'] for a in attacks]
    axes[1].bar(attacks, transfer_ratios, alpha=0.8, color='green')
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='Perfect transfer')
    axes[1].set_xlabel('Attack Method')
    axes[1].set_ylabel('Transferability Ratio')
    axes[1].set_title('Transferability Ratio (Target/Source)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved transferability plot to {save_path}")
    else:
        plt.show()

    plt.close()


def create_results_table(
    results: Dict[str, Dict[str, any]],
    save_path: Optional[str] = None
) -> str:
    """
    Create formatted results table

    Args:
        results: Dictionary of results
        save_path: Path to save table (optional)

    Returns:
        Formatted table string
    """
    # Create markdown table
    lines = []
    lines.append("| Attack | Fooling Rate | L2 Norm | L∞ Norm | Time (s) |")
    lines.append("|--------|--------------|---------|---------|----------|")

    for attack_name, attack_results in results.items():
        fooling_rate = attack_results.get('fooling_rate', 0)
        l2_norm = attack_results.get('l2_norm', 0)
        linf_norm = attack_results.get('linf_norm', 0)
        time_taken = attack_results.get('time', 0)

        lines.append(
            f"| {attack_name} | {fooling_rate:.2f}% | "
            f"{l2_norm:.4f} | {linf_norm:.4f} | {time_taken:.2f} |"
        )

    table_str = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"✓ Saved results table to {save_path}")

    return table_str


def save_results_json(
    results: Dict,
    save_path: str
):
    """
    Save results as JSON

    Args:
        results: Results dictionary
        save_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"✓ Saved results to {save_path}")


def generate_comprehensive_report(
    uap_results: Dict,
    analysis_results: Dict,
    transferability_results: Dict,
    output_dir: str
):
    """
    Generate comprehensive analysis report with all visualizations

    Args:
        uap_results: UAP computation results
        analysis_results: Analysis results (correlation, PCA)
        transferability_results: Transferability results
        output_dir: Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*60}\n")

    # 1. Save perturbation visualization
    print("1. Generating perturbation visualization...")
    if 'perturbation' in uap_results:
        visualize_perturbation(
            perturbation=uap_results['perturbation'],
            save_path=str(output_path / 'uap_perturbation.png'),
            title='Universal Adversarial Perturbation'
        )

    # 2. Save correlation matrix
    print("2. Generating correlation matrix...")
    if 'correlation_matrix' in analysis_results:
        visualize_correlation_matrix(
            correlation_matrix=analysis_results['correlation_matrix'],
            save_path=str(output_path / 'gradient_correlation.png')
        )

    # 3. Save PCA variance plot
    print("3. Generating PCA variance plot...")
    if 'pca_results' in analysis_results:
        pca_res = analysis_results['pca_results']
        visualize_pca_variance(
            explained_variance=pca_res['explained_variance_ratio'],
            cumulative_variance=pca_res['cumulative_variance'],
            effective_dim=pca_res['effective_dimension'],
            save_path=str(output_path / 'pca_variance.png')
        )

    # 4. Save transferability plot
    print("4. Generating transferability plot...")
    if transferability_results:
        visualize_transferability(
            transferability_results=transferability_results,
            save_path=str(output_path / 'transferability.png')
        )

    # 5. Save all results as JSON
    print("5. Saving results JSON...")
    all_results = {
        'uap_results': uap_results,
        'analysis_results': analysis_results,
        'transferability_results': transferability_results,
    }
    save_results_json(all_results, str(output_path / 'all_results.json'))

    print(f"\n✓ Comprehensive report generated in {output_dir}/")
    print(f"  - uap_perturbation.png")
    print(f"  - gradient_correlation.png")
    print(f"  - pca_variance.png")
    print(f"  - transferability.png")
    print(f"  - all_results.json")


if __name__ == "__main__":
    # Test visualization utilities
    print("Testing evaluation utilities...")

    # Create dummy perturbation
    perturbation = torch.randn(3, 32, 32) * 0.01

    # Test perturbation visualization
    print("\n1. Testing perturbation visualization...")
    visualize_perturbation(perturbation, title="Test Perturbation")

    # Test correlation matrix
    print("\n2. Testing correlation matrix...")
    corr_matrix = np.random.rand(50, 50)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1)
    visualize_correlation_matrix(corr_matrix)

    # Test PCA variance
    print("\n3. Testing PCA variance plot...")
    explained_var = np.array([0.3, 0.2, 0.15, 0.1, 0.08] + [0.01] * 15)
    cumulative_var = np.cumsum(explained_var)
    visualize_pca_variance(explained_var, cumulative_var, effective_dim=5)

    print("\n✓ All evaluation tests passed!")
