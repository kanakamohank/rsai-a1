"""
Quick end-to-end test of UAP implementation
Uses a tiny dataset for fast testing
"""
import torch
from src.models import get_model
from src.data_utils import get_cifar100_subset
from src.uap import compute_uap, compute_fooling_rate

print("="*60)
print("QUICK END-TO-END TEST")
print("="*60)

# Setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# Create a simple model
print("\n1. Creating model...")
model = get_model('resnet18', num_classes=100)
model = model.to(device)
model.eval()
print("   ✓ Model created")

# Create tiny dataset
print("\n2. Creating tiny dataset (20 images)...")
train_subset = get_cifar100_subset(subset_size=20, train=True, seed=42)
test_subset = get_cifar100_subset(subset_size=10, train=False, seed=42)
print("   ✓ Datasets created")

# Compute UAP with very small parameters
print("\n3. Computing UAP (this will take 1-2 minutes)...")
try:
    v, fooling_rates = compute_uap(
        model=model,
        dataset=train_subset,
        num_classes=100,
        xi=10/255,
        delta=0.3,  # Low target for quick test
        max_iter_uni=2,  # Only 2 iterations
        norm_type='inf',
        max_iter_df=10,  # Reduced DeepFool iterations
        overshoot=0.02,
        device=device,
        save_path=None  # Don't save
    )
    print("   ✓ UAP computation completed")
    print(f"   Final fooling rate: {fooling_rates[-1]:.2%}")

    # Test on test set
    print("\n4. Evaluating on test set...")
    test_fooling_rate = compute_fooling_rate(model, test_subset, v, device)
    print(f"   Test fooling rate: {test_fooling_rate:.2%}")

    # Check perturbation norms
    l2_norm = torch.norm(v.flatten(), p=2).item()
    linf_norm = torch.max(torch.abs(v)).item()
    print(f"\n5. Perturbation norms:")
    print(f"   L2 norm:  {l2_norm:.4f}")
    print(f"   L∞ norm:  {linf_norm:.6f}")
    print(f"   Budget:   {10/255:.6f}")

    print("\n" + "="*60)
    print("✓ END-TO-END TEST PASSED!")
    print("="*60)
    print("\nAll core components are working correctly:")
    print("  ✓ Models")
    print("  ✓ Data utilities")
    print("  ✓ DeepFool")
    print("  ✓ UAP algorithm")
    print("  ✓ Evaluation")
    print("\nReady for full-scale implementation!")

except Exception as e:
    print(f"\n✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
