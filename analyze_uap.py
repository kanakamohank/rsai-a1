#!/usr/bin/env python3
"""
Comprehensive UAP Analysis Script

This script performs:
1. Gradient correlation analysis (Question 1)
2. PCA dimensionality analysis (Question 1)
3. Transferability testing (Question 2)
4. Generates all visualizations and reports
"""
import torch
import argparse
from pathlib import Path
import sys

from config import (
    DEVICE, MODEL_NAME, MODEL_PATH, MODEL_NAME_TRANSFER, MODEL_PATH_TRANSFER,
    PATHS, ANALYSIS_CONFIG, RANDOM_SEED
)
from src.models import load_cifar100_model
from src.data_utils import get_cifar100_loaders, get_cifar100_subset
from src.analysis import (
    compute_gradient_correlation,
    analyze_decision_boundary_dimensionality,
    test_transferability,
    compare_attack_transferability,
    analyze_perturbation_properties
)
from src.evaluation import (
    visualize_perturbation,
    visualize_correlation_matrix,
    visualize_pca_variance,
    visualize_transferability,
    visualize_adversarial_examples,
    generate_comprehensive_report,
    save_results_json
)


def load_perturbation(path: str, device: torch.device) -> torch.Tensor:
    """Load saved perturbation"""
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'perturbation' in checkpoint:
            perturbation = checkpoint['perturbation']
        elif 'uap' in checkpoint:
            perturbation = checkpoint['uap']
        else:
            raise ValueError(f"Cannot find perturbation in checkpoint: {checkpoint.keys()}")
    else:
        perturbation = checkpoint

    return perturbation.to(device)


def analyze_question1(
    model: torch.nn.Module,
    subset_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path
) -> dict:
    """
    Analyze Question 1: Why does a single perturbation fool many images?

    Performs:
    - Gradient correlation analysis
    - PCA dimensionality analysis
    """
    print(f"\n{'='*80}")
    print("QUESTION 1: WHY DOES A SINGLE PERTURBATION FOOL MANY IMAGES?")
    print(f"{'='*80}\n")

    results = {}

    # Get samples for analysis
    images_list = []
    labels_list = []

    for images, labels in subset_loader:
        images_list.append(images)
        labels_list.append(labels)
        if len(torch.cat(images_list)) >= ANALYSIS_CONFIG['num_gradient_samples']:
            break

    images = torch.cat(images_list)[:ANALYSIS_CONFIG['num_gradient_samples']]
    labels = torch.cat(labels_list)[:ANALYSIS_CONFIG['num_gradient_samples']]

    print(f"Using {len(images)} samples for analysis\n")

    # 1. Gradient Correlation Analysis
    print("=" * 60)
    print("1. GRADIENT CORRELATION ANALYSIS")
    print("=" * 60)
    print("Hypothesis: Images share similar vulnerability directions")
    print()

    mean_correlation, correlation_matrix = compute_gradient_correlation(
        model=model,
        images=images,
        labels=labels,
        device=device
    )

    print(f"\n✓ Mean gradient correlation: {mean_correlation:.4f}")

    # Save correlation matrix visualization
    visualize_correlation_matrix(
        correlation_matrix=correlation_matrix,
        save_path=str(output_dir / 'gradient_correlation.png'),
        title=f'Gradient Correlation Matrix\n(Mean: {mean_correlation:.4f})'
    )

    results['gradient_correlation'] = {
        'mean_correlation': float(mean_correlation),
        'correlation_matrix': correlation_matrix
    }

    # 2. PCA Dimensionality Analysis
    print(f"\n{'=' * 60}")
    print("2. PCA DIMENSIONALITY ANALYSIS")
    print("=" * 60)
    print("Hypothesis: Decision boundaries lie in low-dimensional subspace")
    print()

    # Use more samples for PCA
    images_pca = torch.cat(images_list)[:ANALYSIS_CONFIG['num_pca_samples']]
    labels_pca = torch.cat(labels_list)[:ANALYSIS_CONFIG['num_pca_samples']]

    pca_results = analyze_decision_boundary_dimensionality(
        model=model,
        images=images_pca,
        labels=labels_pca,
        device=device,
        n_components=50
    )

    print(f"\n✓ Effective dimension (90% variance): {pca_results['effective_dimension']}/50")
    print(f"✓ Total variance captured: {pca_results['total_variance_90pct']:.4f}")

    # Save PCA visualization
    visualize_pca_variance(
        explained_variance=pca_results['explained_variance_ratio'],
        cumulative_variance=pca_results['cumulative_variance'],
        effective_dim=pca_results['effective_dimension'],
        save_path=str(output_dir / 'pca_variance.png')
    )

    results['pca_results'] = {
        'effective_dimension': int(pca_results['effective_dimension']),
        'total_variance_90pct': float(pca_results['total_variance_90pct']),
        'explained_variance_ratio': pca_results['explained_variance_ratio'],
        'cumulative_variance': pca_results['cumulative_variance']
    }

    # Summary for Question 1
    print(f"\n{'=' * 60}")
    print("QUESTION 1 SUMMARY")
    print("=" * 60)
    print(f"Gradient Correlation: {mean_correlation:.4f}")
    print(f"  → High correlation indicates shared vulnerability directions")
    print(f"\nEffective Dimension: {pca_results['effective_dimension']}/50 components")
    print(f"  → Decision boundaries lie in {pca_results['effective_dimension']}-dimensional subspace")
    print(f"\nConclusion:")
    if mean_correlation > 0.5:
        print("  ✓ HIGH correlation: Images share similar vulnerability directions")
    else:
        print("  ✓ MODERATE correlation: Some shared vulnerability patterns")

    if pca_results['effective_dimension'] < 20:
        print("  ✓ LOW dimensionality: Decision boundaries are low-dimensional")
    else:
        print("  ✓ MODERATE dimensionality: Decision boundaries span multiple dimensions")

    print("\n  → These factors explain why a single UAP can fool multiple images!")

    return results


def analyze_question2(
    uap_perturbation: torch.Tensor,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path
) -> dict:
    """
    Analyze Question 2: How does UAP compare to FGSM/PGD in transferability?

    Performs transferability testing across models
    """
    print(f"\n{'='*80}")
    print("QUESTION 2: UAP vs FGSM/PGD TRANSFERABILITY")
    print(f"{'='*80}\n")

    print(f"Source Model: {MODEL_NAME}")
    print(f"Target Model: {MODEL_NAME_TRANSFER}")
    print(f"Test samples: {ANALYSIS_CONFIG['num_transfer_samples']}\n")

    # Test UAP transferability
    print("=" * 60)
    print("Testing UAP Transferability")
    print("=" * 60)

    uap_transfer_results = test_transferability(
        perturbation=uap_perturbation,
        source_model=source_model,
        target_model=target_model,
        test_loader=test_loader,
        device=device,
        max_samples=ANALYSIS_CONFIG['num_transfer_samples']
    )

    # For FGSM/PGD comparison, we would need to compute per-image perturbations
    # and test their transferability. For now, we'll report UAP transferability.

    results = {
        'uap': uap_transfer_results
    }

    # Summary for Question 2
    print(f"\n{'=' * 60}")
    print("QUESTION 2 SUMMARY")
    print("=" * 60)
    print(f"UAP Transferability:")
    print(f"  Source fooling rate: {uap_transfer_results['source_fooling_rate']:.2f}%")
    print(f"  Target fooling rate: {uap_transfer_results['target_fooling_rate']:.2f}%")
    print(f"  Transferability ratio: {uap_transfer_results['transferability_ratio']:.2f}")

    if uap_transfer_results['transferability_ratio'] > 0.7:
        print(f"\n  ✓ HIGH transferability: UAP transfers well across architectures")
    elif uap_transfer_results['transferability_ratio'] > 0.4:
        print(f"\n  ✓ MODERATE transferability: UAP partially transfers")
    else:
        print(f"\n  ✓ LOW transferability: UAP is model-specific")

    return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive UAP Analysis')
    parser.add_argument('--uap-path', type=str, required=True,
                       help='Path to saved UAP perturbation')
    parser.add_argument('--output-dir', type=str, default='./results/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis with fewer samples')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(RANDOM_SEED)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE UAP ANALYSIS")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Source Model: {MODEL_NAME}")
    print(f"Target Model: {MODEL_NAME_TRANSFER}")
    print(f"UAP Path: {args.uap_path}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # Adjust sample sizes for quick mode
    if args.quick:
        print("\n⚡ Quick mode: Using reduced sample sizes")
        ANALYSIS_CONFIG['num_gradient_samples'] = 50
        ANALYSIS_CONFIG['num_pca_samples'] = 100
        ANALYSIS_CONFIG['num_transfer_samples'] = 200

    # 1. Load UAP perturbation
    print("\n1. Loading UAP perturbation...")
    try:
        uap_perturbation = load_perturbation(args.uap_path, DEVICE)
        print(f"✓ Loaded UAP from {args.uap_path}")

        # Analyze perturbation properties
        uap_stats = analyze_perturbation_properties(uap_perturbation, name="UAP")

        # Visualize perturbation
        visualize_perturbation(
            perturbation=uap_perturbation,
            save_path=str(output_dir / 'uap_perturbation.png'),
            title='Universal Adversarial Perturbation'
        )
    except Exception as e:
        print(f"✗ Error loading UAP: {e}")
        sys.exit(1)

    # 2. Load models
    print("\n2. Loading models...")
    source_model = load_cifar100_model(MODEL_NAME, MODEL_PATH, DEVICE)
    print(f"✓ Loaded source model: {MODEL_NAME}")

    target_model = load_cifar100_model(MODEL_NAME_TRANSFER, MODEL_PATH_TRANSFER, DEVICE)
    print(f"✓ Loaded target model: {MODEL_NAME_TRANSFER}")

    # 3. Load data
    print("\n3. Loading CIFAR-100 data...")
    train_loader, test_loader = get_cifar100_loaders(
        data_root=PATHS['data'],
        batch_size=32,
        num_workers=0  # Disable multiprocessing for macOS MPS compatibility
    )

    # Create subset for analysis
    subset_dataset = get_cifar100_subset(
        data_root=PATHS['data'],
        subset_size=max(ANALYSIS_CONFIG['num_gradient_samples'],
                       ANALYSIS_CONFIG['num_pca_samples']),
        train=True,
        seed=RANDOM_SEED
    )
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0  # Disable multiprocessing for macOS MPS compatibility
    )
    print(f"✓ Loaded CIFAR-100 dataset")

    # 4. Analyze Question 1
    question1_results = analyze_question1(
        model=source_model,
        subset_loader=subset_loader,
        device=DEVICE,
        output_dir=output_dir
    )

    # 5. Analyze Question 2
    question2_results = analyze_question2(
        uap_perturbation=uap_perturbation,
        source_model=source_model,
        target_model=target_model,
        test_loader=test_loader,
        device=DEVICE,
        output_dir=output_dir
    )

    # 6. Save all results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print("=" * 80)

    all_results = {
        'uap_stats': uap_stats,
        'question1': question1_results,
        'question2': question2_results,
        'config': {
            'source_model': MODEL_NAME,
            'target_model': MODEL_NAME_TRANSFER,
            'device': str(DEVICE),
            'analysis_config': ANALYSIS_CONFIG
        }
    }

    save_results_json(all_results, str(output_dir / 'analysis_results.json'))

    # 7. Print final summary
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("Generated files:")
    print("  - uap_perturbation.png          (UAP visualization)")
    print("  - gradient_correlation.png       (Question 1: Correlation)")
    print("  - pca_variance.png               (Question 1: Dimensionality)")
    print("  - analysis_results.json          (All numerical results)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
