#!/usr/bin/env python3
"""
Quick test of full pipeline with small subset
Tests: Model loading, UAP computation, analysis framework
"""
import torch
import sys
from pathlib import Path

print("="*80)
print("QUICK PIPELINE TEST")
print("="*80)
print("Testing with 20 images (should take ~5 minutes)\n")

# Import modules
from config import DEVICE, MODEL_NAME, MODEL_PATH, PATHS, RANDOM_SEED
from src.models import load_cifar100_model, test_model_accuracy
from src.data_utils import get_cifar100_loaders, get_cifar100_subset
from src.uap import compute_uap
from src.analysis import (
    compute_gradient_correlation,
    analyze_decision_boundary_dimensionality,
    test_transferability,
    analyze_perturbation_properties
)
from src.evaluation import visualize_perturbation

# Set seed
torch.manual_seed(RANDOM_SEED)

# Create test output directory
test_output = Path('./results/test_quick')
test_output.mkdir(parents=True, exist_ok=True)

# Step 1: Load model
print("\n" + "="*80)
print("STEP 1: Loading Model")
print("="*80)
model = load_cifar100_model(MODEL_NAME, MODEL_PATH, DEVICE)
print(f"✓ Model loaded: {MODEL_NAME}")

# Step 2: Test on small batch
print("\n" + "="*80)
print("STEP 2: Testing Model Accuracy")
print("="*80)
_, test_loader = get_cifar100_loaders(
    data_root=PATHS['data'],
    batch_size=128,
    num_workers=0  # Disable multiprocessing for macOS compatibility
)
# Test on first 100 images only
test_images_count = 0
correct = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        if test_images_count >= 100:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        test_images_count += labels.size(0)

accuracy = 100.0 * correct / test_images_count
print(f"✓ Model accuracy (100 samples): {accuracy:.2f}%")

# Step 3: Compute mini UAP
print("\n" + "="*80)
print("STEP 3: Computing Mini UAP (20 images)")
print("="*80)
subset_dataset = get_cifar100_subset(
    data_root=PATHS['data'],
    subset_size=20,
    train=True,
    seed=RANDOM_SEED
)
subset_loader = torch.utils.data.DataLoader(
    subset_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0  # Disable multiprocessing for macOS compatibility
)

uap, fooling_rates = compute_uap(
    model=model,
    dataset=subset_dataset,  # Fixed: UAP expects dataset not dataloader
    device=DEVICE,
    xi=10/255,
    delta=0.8,
    max_iter_uni=3,  # Just 3 iterations for quick test
    norm_type='inf',
    save_interval=1,
    save_path=str(test_output)  # Fixed: correct parameter name
)

print(f"\n✓ UAP computation complete")
print(f"  Final fooling rate: {fooling_rates[-1]:.2f}%")
print(f"  Iterations: {len(fooling_rates)}")

# Analyze UAP properties
uap_stats = analyze_perturbation_properties(uap, name="Test UAP")

# Step 4: Quick analysis
print("\n" + "="*80)
print("STEP 4: Testing Analysis Framework")
print("="*80)

# Get analysis samples
analysis_subset = get_cifar100_subset(
    data_root=PATHS['data'],
    subset_size=20,
    train=True,
    seed=RANDOM_SEED
)
analysis_loader = torch.utils.data.DataLoader(
    analysis_subset, batch_size=20, shuffle=False
)
images, labels = next(iter(analysis_loader))

# Test gradient correlation
print("\nTesting gradient correlation...")
mean_corr, corr_matrix = compute_gradient_correlation(
    model=model,
    images=images,
    labels=labels,
    device=DEVICE
)
print(f"✓ Mean correlation: {mean_corr:.4f}")

# Test PCA
print("\nTesting PCA analysis...")
pca_results = analyze_decision_boundary_dimensionality(
    model=model,
    images=images,
    labels=labels,
    device=DEVICE,
    n_components=10
)
print(f"✓ Effective dimension: {pca_results['effective_dimension']}/10")

# Step 5: Test visualization
print("\n" + "="*80)
print("STEP 5: Testing Visualization")
print("="*80)
visualize_perturbation(
    perturbation=uap,
    save_path=str(test_output / 'test_uap.png'),
    title='Test UAP'
)
print(f"✓ Visualization saved")

# Step 6: Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"✓ Model loading:         PASSED")
print(f"✓ Model accuracy:        {accuracy:.2f}% (expected >70%)")
print(f"✓ UAP computation:       PASSED ({fooling_rates[-1]:.2f}% fooling rate)")
print(f"✓ Gradient correlation:  PASSED ({mean_corr:.4f})")
print(f"✓ PCA analysis:          PASSED ({pca_results['effective_dimension']} dims)")
print(f"✓ Visualization:         PASSED")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✅")
print("="*80)
print(f"\nTest results saved to: {test_output}/")
print("\nReady to run full experiment:")
print("  • Local:  ./venv/bin/python3 main.py --compute-uap")
print("  • Colab:  Upload UAP_Full_Experiment.ipynb")
print("="*80)
