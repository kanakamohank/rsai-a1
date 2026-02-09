"""
Main script for UAP project

Usage:
    python main.py --test-model       # Test model accuracy
    python main.py --compute-uap      # Compute UAP
    python main.py --evaluate         # Evaluate UAP on test set
"""
import torch
import argparse
import os

from config import *
from src.models import load_cifar100_model, test_model_accuracy
from src.data_utils import get_cifar100_loaders, get_cifar100_subset
from src.uap import compute_uap, load_uap, compute_fooling_rate


def test_model(args):
    """Test model accuracy on CIFAR-100"""
    print("\n" + "="*80)
    print("TESTING MODEL ACCURACY")
    print("="*80)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = load_cifar100_model(MODEL_NAME, MODEL_PATH, DEVICE)

    # Load test data
    print("\nLoading CIFAR-100 test set...")
    _, test_loader = get_cifar100_loaders(
        data_root=PATHS['data'],
        batch_size=128
    )

    # Test accuracy
    print("\nEvaluating model on test set...")
    accuracy = test_model_accuracy(model, test_loader, DEVICE)

    print(f"\n{'='*80}")
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")

    if accuracy < 50:
        print("⚠ WARNING: Model accuracy is very low!")
        print("  Consider training a better model or using pre-trained weights.")
    elif accuracy < 70:
        print("⚠ WARNING: Model accuracy is below 70%")
        print("  UAP results may be less reliable.")
    else:
        print("✓ Model accuracy is good!")


def compute_uap_perturbation(args):
    """Compute Universal Adversarial Perturbation"""
    print("\n" + "="*80)
    print("COMPUTING UNIVERSAL ADVERSARIAL PERTURBATION")
    print("="*80)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = load_cifar100_model(MODEL_NAME, MODEL_PATH, DEVICE)

    # Load training subset
    print(f"\nLoading training subset ({UAP_CONFIG['subset_size']} images)...")
    train_subset = get_cifar100_subset(
        data_root=PATHS['data'],
        subset_size=UAP_CONFIG['subset_size'],
        train=True,
        seed=RANDOM_SEED
    )

    # Create results directory
    os.makedirs(PATHS['results'], exist_ok=True)
    save_path = os.path.join(PATHS['results'], 'uap_perturbation.pt')

    # Compute UAP
    print("\nStarting UAP computation...")
    v, fooling_rates = compute_uap(
        model=model,
        dataset=train_subset,
        num_classes=NUM_CLASSES,
        xi=UAP_CONFIG['xi'],
        delta=UAP_CONFIG['delta'],
        max_iter_uni=UAP_CONFIG['max_iter_uni'],
        norm_type=UAP_CONFIG['norm_type'],
        max_iter_df=DEEPFOOL_CONFIG['max_iter'],
        overshoot=DEEPFOOL_CONFIG['overshoot'],
        device=DEVICE,
        save_path=save_path,
        save_interval=UAP_CONFIG['save_interval']
    )

    print(f"\n✓ UAP computation complete!")
    print(f"  Final fooling rate: {fooling_rates[-1]:.2%}")
    print(f"  Perturbation saved to: {save_path}")


def evaluate_uap(args):
    """Evaluate UAP on test set"""
    print("\n" + "="*80)
    print("EVALUATING UNIVERSAL ADVERSARIAL PERTURBATION")
    print("="*80)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = load_cifar100_model(MODEL_NAME, MODEL_PATH, DEVICE)

    # Load UAP
    uap_path = args.uap_path or os.path.join(PATHS['results'], 'uap_perturbation.pt')
    print(f"\nLoading UAP from: {uap_path}")

    if not os.path.exists(uap_path):
        print(f"✗ Error: UAP file not found at {uap_path}")
        print("  Run with --compute-uap first to generate the perturbation.")
        return

    v = load_uap(uap_path, DEVICE)

    # Load test data
    print("\nLoading CIFAR-100 test set...")
    _, test_loader = get_cifar100_loaders(
        data_root=PATHS['data'],
        batch_size=128
    )

    # Get test dataset (without dataloader for fooling rate computation)
    from torchvision import datasets, transforms
    mean = CIFAR100_MEAN
    std = CIFAR100_STD
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_dataset = datasets.CIFAR100(
        root=PATHS['data'],
        train=False,
        download=True,
        transform=transform
    )

    # Evaluate clean accuracy
    print("\nEvaluating clean model accuracy...")
    clean_accuracy = test_model_accuracy(model, test_loader, DEVICE)

    # Evaluate fooling rate
    print("\nComputing fooling rate on test set...")
    fooling_rate = compute_fooling_rate(
        model=model,
        dataset=test_dataset,
        perturbation=v,
        device=DEVICE,
        batch_size=128
    )

    # Compute perturbation norms
    l2_norm = torch.norm(v.flatten(), p=2).item()
    linf_norm = torch.max(torch.abs(v)).item()

    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Clean Model Accuracy:     {clean_accuracy:.2f}%")
    print(f"Fooling Rate:             {fooling_rate:.2%} ({fooling_rate*100:.2f}%)")
    print(f"Perturbation L2 Norm:     {l2_norm:.6f}")
    print(f"Perturbation L∞ Norm:     {linf_norm:.6f}")
    print(f"Perturbation Budget (ξ):  {UAP_CONFIG['xi']:.6f}")
    print(f"{'='*80}\n")

    if fooling_rate >= 0.70:
        print("✓ SUCCESS: Fooling rate ≥ 70%")
    else:
        print("⚠ WARNING: Fooling rate < 70%")
        print("  Consider:")
        print("  - Increasing perturbation budget (xi)")
        print("  - Increasing max_iter_uni")
        print("  - Using more training images")


def main():
    parser = argparse.ArgumentParser(description='Universal Adversarial Perturbations')

    # Actions
    parser.add_argument('--test-model', action='store_true',
                       help='Test model accuracy on CIFAR-100')
    parser.add_argument('--compute-uap', action='store_true',
                       help='Compute universal adversarial perturbation')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate UAP on test set')

    # Optional arguments
    parser.add_argument('--uap-path', type=str, default=None,
                       help='Path to saved UAP (for evaluation)')
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                       help='Model architecture (resnet18 or vgg16)')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                       help='Path to model checkpoint')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    print(f"\n{'='*80}")
    print("Universal Adversarial Perturbations - CIFAR-100")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"{'='*80}\n")

    # Execute requested action
    if args.test_model:
        test_model(args)
    elif args.compute_uap:
        compute_uap_perturbation(args)
    elif args.evaluate:
        evaluate_uap(args)
    else:
        parser.print_help()
        print("\nNo action specified. Use --help to see available options.")
        print("\nQuick start:")
        print("  1. python main.py --test-model      # Check model accuracy")
        print("  2. python main.py --compute-uap     # Compute UAP")
        print("  3. python main.py --evaluate        # Evaluate on test set")


if __name__ == "__main__":
    main()
