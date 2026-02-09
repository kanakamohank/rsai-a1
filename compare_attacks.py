"""
Compare UAP vs FGSM vs PGD attacks

This script evaluates all three attack methods on CIFAR-100 test set
and generates comparison metrics.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import argparse

from config import *
from src.models import load_cifar100_model
from src.data_utils import get_cifar100_subset
from src.uap import load_uap, compute_fooling_rate
from src.fgsm import fgsm_batch
from src.pgd import pgd_batch


def evaluate_uap(model, test_dataset, uap_path, device):
    """Evaluate UAP attack"""
    print("\n" + "="*80)
    print("EVALUATING UAP ATTACK")
    print("="*80)

    # Load UAP
    v = load_uap(uap_path, device)

    # Compute fooling rate
    start_time = time.time()
    fooling_rate = compute_fooling_rate(model, test_dataset, v, device, batch_size=100)
    elapsed_time = time.time() - start_time

    # Compute perturbation norms
    l2_norm = torch.norm(v.flatten(), p=2).item()
    linf_norm = torch.max(torch.abs(v)).item()

    results = {
        'attack': 'UAP',
        'fooling_rate': fooling_rate * 100,
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'time': elapsed_time,
        'perturbations': 1,  # Single universal perturbation
    }

    print(f"Fooling rate: {results['fooling_rate']:.2f}%")
    print(f"L2 norm: {results['l2_norm']:.4f}")
    print(f"L∞ norm: {results['linf_norm']:.6f}")
    print(f"Time: {results['time']:.2f} seconds")

    return results


def evaluate_fgsm(model, test_dataset, epsilon, device, num_samples=None):
    """Evaluate FGSM attack"""
    print("\n" + "="*80)
    print("EVALUATING FGSM ATTACK")
    print("="*80)

    model.eval()

    # Create dataloader
    if num_samples:
        indices = list(range(min(num_samples, len(test_dataset))))
        subset = torch.utils.data.Subset(test_dataset, indices)
    else:
        subset = test_dataset

    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=100, shuffle=False, num_workers=0
    )

    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    total_l2_norm = 0
    total_linf_norm = 0

    start_time = time.time()

    for images, labels in tqdm(dataloader, desc="FGSM"):
        images, labels = images.to(device), labels.to(device)

        # Get clean predictions
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            total_correct_clean += (clean_preds == labels).sum().item()

        # Apply FGSM
        adv_images, adv_preds, _ = fgsm_batch(model, images, labels, epsilon)

        # Count correct adversarial predictions
        total_correct_adv += (adv_preds == labels).sum().item()

        # Compute perturbation norms
        perturbations = adv_images - images
        total_l2_norm += torch.norm(perturbations.flatten(1), p=2, dim=1).sum().item()
        total_linf_norm += torch.max(torch.abs(perturbations.flatten(1)), dim=1)[0].sum().item()

        total_samples += images.size(0)

    elapsed_time = time.time() - start_time

    clean_accuracy = 100.0 * total_correct_clean / total_samples
    adv_accuracy = 100.0 * total_correct_adv / total_samples
    fooling_rate = 100.0 - adv_accuracy

    avg_l2_norm = total_l2_norm / total_samples
    avg_linf_norm = total_linf_norm / total_samples

    results = {
        'attack': 'FGSM',
        'clean_accuracy': clean_accuracy,
        'adv_accuracy': adv_accuracy,
        'fooling_rate': fooling_rate,
        'l2_norm': avg_l2_norm,
        'linf_norm': avg_linf_norm,
        'time': elapsed_time,
        'perturbations': total_samples,
    }

    print(f"Clean accuracy: {results['clean_accuracy']:.2f}%")
    print(f"Adversarial accuracy: {results['adv_accuracy']:.2f}%")
    print(f"Fooling rate: {results['fooling_rate']:.2f}%")
    print(f"Avg L2 norm: {results['l2_norm']:.4f}")
    print(f"Avg L∞ norm: {results['linf_norm']:.6f}")
    print(f"Time: {results['time']:.2f} seconds")
    print(f"Perturbations computed: {results['perturbations']}")

    return results


def evaluate_pgd(model, test_dataset, epsilon, alpha, num_iter, device, num_samples=None):
    """Evaluate PGD attack"""
    print("\n" + "="*80)
    print("EVALUATING PGD ATTACK")
    print("="*80)

    model.eval()

    # Create dataloader
    if num_samples:
        indices = list(range(min(num_samples, len(test_dataset))))
        subset = torch.utils.data.Subset(test_dataset, indices)
    else:
        subset = test_dataset

    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=100, shuffle=False, num_workers=0
    )

    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    total_l2_norm = 0
    total_linf_norm = 0

    start_time = time.time()

    for images, labels in tqdm(dataloader, desc="PGD"):
        images, labels = images.to(device), labels.to(device)

        # Get clean predictions
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            total_correct_clean += (clean_preds == labels).sum().item()

        # Apply PGD
        adv_images, adv_preds, _ = pgd_batch(
            model, images, labels, epsilon, alpha, num_iter
        )

        # Count correct adversarial predictions
        total_correct_adv += (adv_preds == labels).sum().item()

        # Compute perturbation norms
        perturbations = adv_images - images
        total_l2_norm += torch.norm(perturbations.flatten(1), p=2, dim=1).sum().item()
        total_linf_norm += torch.max(torch.abs(perturbations.flatten(1)), dim=1)[0].sum().item()

        total_samples += images.size(0)

    elapsed_time = time.time() - start_time

    clean_accuracy = 100.0 * total_correct_clean / total_samples
    adv_accuracy = 100.0 * total_correct_adv / total_samples
    fooling_rate = 100.0 - adv_accuracy

    avg_l2_norm = total_l2_norm / total_samples
    avg_linf_norm = total_linf_norm / total_samples

    results = {
        'attack': 'PGD',
        'clean_accuracy': clean_accuracy,
        'adv_accuracy': adv_accuracy,
        'fooling_rate': fooling_rate,
        'l2_norm': avg_l2_norm,
        'linf_norm': avg_linf_norm,
        'time': elapsed_time,
        'perturbations': total_samples,
    }

    print(f"Clean accuracy: {results['clean_accuracy']:.2f}%")
    print(f"Adversarial accuracy: {results['adv_accuracy']:.2f}%")
    print(f"Fooling rate: {results['fooling_rate']:.2f}%")
    print(f"Avg L2 norm: {results['l2_norm']:.4f}")
    print(f"Avg L∞ norm: {results['linf_norm']:.6f}")
    print(f"Time: {results['time']:.2f} seconds")
    print(f"Perturbations computed: {results['perturbations']}")

    return results


def print_comparison(results_list):
    """Print comparison table"""
    print("\n" + "="*80)
    print("ATTACK COMPARISON SUMMARY")
    print("="*80)

    # Print table header
    print(f"{'Attack':<10} {'Fooling Rate':<15} {'Avg L2':<12} {'Avg L∞':<12} {'Time (s)':<12} {'#Perturb':<10}")
    print("-" * 80)

    # Print results
    for r in results_list:
        print(f"{r['attack']:<10} {r['fooling_rate']:>13.2f}% "
              f"{r['l2_norm']:>11.4f} {r['linf_norm']:>11.6f} "
              f"{r['time']:>11.2f} {r['perturbations']:>9d}")

    print("=" * 80)

    # Analysis
    print("\nKEY OBSERVATIONS:")
    print("-" * 80)

    # Find best attack by fooling rate
    best_attack = max(results_list, key=lambda x: x['fooling_rate'])
    print(f"• Highest fooling rate: {best_attack['attack']} ({best_attack['fooling_rate']:.2f}%)")

    # Compare efficiency (fooling rate per perturbation)
    for r in results_list:
        if r['perturbations'] > 0:
            efficiency = r['fooling_rate'] / r['perturbations']
            print(f"• {r['attack']} efficiency: {efficiency:.6f}% per perturbation")

    # UAP advantage
    uap_result = next((r for r in results_list if r['attack'] == 'UAP'), None)
    if uap_result:
        print(f"\n• UAP uses only {uap_result['perturbations']} perturbation for entire dataset")
        print(f"  vs {results_list[1]['perturbations']} perturbations for per-image attacks")


def main():
    parser = argparse.ArgumentParser(description='Compare UAP vs FGSM vs PGD')
    parser.add_argument('--uap-path', type=str, default='results/uap_perturbation.pt',
                       help='Path to UAP file')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of test samples to use (None = all)')
    parser.add_argument('--skip-uap', action='store_true',
                       help='Skip UAP evaluation')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("ATTACK COMPARISON: UAP vs FGSM vs PGD")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epsilon: {FGSM_CONFIG['epsilon']:.6f}")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model = load_cifar100_model(MODEL_NAME, MODEL_PATH, DEVICE)

    # Load test dataset
    print("Loading CIFAR-100 test set...")
    if args.num_samples:
        test_dataset = get_cifar100_subset(
            data_root=PATHS['data'],
            subset_size=args.num_samples,
            train=False,
            seed=RANDOM_SEED
        )
    else:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])
        test_dataset = datasets.CIFAR100(
            root=PATHS['data'],
            train=False,
            download=True,
            transform=transform
        )

    print(f"Test dataset size: {len(test_dataset)}")

    # Evaluate attacks
    results = []

    # UAP
    if not args.skip_uap:
        try:
            uap_results = evaluate_uap(model, test_dataset, args.uap_path, DEVICE)
            results.append(uap_results)
        except FileNotFoundError:
            print(f"\n⚠ UAP file not found: {args.uap_path}")
            print("  Run 'python main.py --compute-uap' first or use --skip-uap")

    # FGSM
    fgsm_results = evaluate_fgsm(
        model, test_dataset, FGSM_CONFIG['epsilon'], DEVICE, args.num_samples
    )
    results.append(fgsm_results)

    # PGD
    pgd_results = evaluate_pgd(
        model, test_dataset,
        PGD_CONFIG['epsilon'],
        PGD_CONFIG['alpha'],
        PGD_CONFIG['num_iter'],
        DEVICE,
        args.num_samples
    )
    results.append(pgd_results)

    # Print comparison
    if results:
        print_comparison(results)

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
