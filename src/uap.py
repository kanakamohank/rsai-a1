"""
Universal Adversarial Perturbation (UAP) algorithm implementation

Reference: Moosavi-Dezfooli et al. (2017)
"Universal adversarial perturbations"
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, List
import os

from .deepfool import deepfool


def project_perturbation(perturbation: torch.Tensor, xi: float, norm_type: str = 'inf') -> torch.Tensor:
    """
    Project perturbation to satisfy norm constraint

    Args:
        perturbation: Perturbation tensor
        xi: Maximum allowed perturbation magnitude
        norm_type: Type of norm ('inf' or 2)

    Returns:
        Projected perturbation
    """
    if norm_type == 'inf':
        # L∞ projection: clip each element
        perturbation = torch.clamp(perturbation, -xi, xi)
    elif norm_type == '2' or norm_type == 2:
        # L2 projection: scale if norm exceeds xi
        norm = torch.norm(perturbation.flatten(), p=2)
        if norm > xi:
            perturbation = perturbation * (xi / norm)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    return perturbation


def compute_fooling_rate(model: nn.Module,
                        dataset: torch.utils.data.Dataset,
                        perturbation: torch.Tensor,
                        device: torch.device,
                        batch_size: int = 100) -> float:
    """
    Compute fooling rate of a universal perturbation

    Args:
        model: Neural network model
        dataset: Dataset to test on
        perturbation: Universal perturbation
        device: Device to run on
        batch_size: Batch size for evaluation

    Returns:
        Fooling rate (0-1)
    """
    model.eval()
    perturbation = perturbation.to(device)

    fooled_count = 0
    total_count = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Original predictions
            orig_outputs = model(images)
            orig_preds = orig_outputs.argmax(dim=1)

            # Perturbed predictions
            # Note: Do not clamp normalized CIFAR images to [0,1] as they have values outside this range
            pert_images = images + perturbation
            pert_outputs = model(pert_images)
            pert_preds = pert_outputs.argmax(dim=1)

            # Count fooled images
            fooled = (orig_preds != pert_preds).sum().item()
            fooled_count += fooled
            total_count += images.size(0)

    fooling_rate = fooled_count / total_count
    return fooling_rate


def compute_uap(model: nn.Module,
                dataset: torch.utils.data.Dataset,
                num_classes: int = 100,
                xi: float = 25/255,
                delta: float = 0.6,
                max_iter_uni: int = 3,
                norm_type: str = 'inf',
                max_iter_df: int = 10,
                overshoot: float = 0.02,
                device: torch.device = torch.device('cpu'),
                save_path: Optional[str] = None,
                save_interval: int = 1) -> Tuple[torch.Tensor, List[float]]:
    """
    Compute Universal Adversarial Perturbation using the algorithm from
    Moosavi-Dezfooli et al. (2017)

    Args:
        model: Neural network model
        dataset: Training dataset (subset)
        num_classes: Number of classes
        xi: Perturbation magnitude constraint
        delta: Target fooling rate
        max_iter_uni: Maximum UAP iterations
        norm_type: Type of norm constraint ('inf' or 2)
        max_iter_df: Maximum DeepFool iterations
        overshoot: DeepFool overshoot parameter
        device: Device to run on
        save_path: Path to save intermediate perturbations
        save_interval: Save every N iterations

    Returns:
        v: Universal perturbation
        fooling_rates: Fooling rate at each iteration
    """
    model.eval()
    model = model.to(device)

    # Initialize universal perturbation
    sample_image, _ = dataset[0]
    v = torch.zeros_like(sample_image).to(device)

    # Track fooling rates
    fooling_rates = []

    # Create dataloader - UAP algorithm requires batch_size=1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            shuffle=True, num_workers=0)

    print(f"\n{'='*80}")
    print(f"Computing Universal Adversarial Perturbation")
    print(f"{'='*80}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Perturbation budget (ξ): {xi:.4f}")
    print(f"Target fooling rate (δ): {delta:.2%}")
    print(f"Norm type: L{norm_type}")
    print(f"{'='*80}\n")

    # Main UAP loop
    for iteration in range(max_iter_uni):
        print(f"\n--- Iteration {iteration + 1}/{max_iter_uni} ---")

        fooled_count = 0
        total_count = 0

        # Process each image in the dataset
        for batch_idx, (image, label) in enumerate(tqdm(dataloader,
                                                        desc=f"Iter {iteration + 1}",
                                                        leave=False)):
            image = image.to(device)
            label = label.to(device)

            # Apply current universal perturbation
            # Note: Do not clamp normalized CIFAR images to [0,1] as they have values outside this range
            pert_image = image + v

            # Check if image is already fooled
            with torch.no_grad():
                orig_output = model(image)
                orig_pred = orig_output.argmax(dim=1).item()

                pert_output = model(pert_image)
                pert_pred = pert_output.argmax(dim=1).item()

            # If not fooled, compute perturbation with DeepFool
            if orig_pred == pert_pred:
                # Apply DeepFool to the perturbed image
                # pert_image is (1, C, H, W), squeeze to (C, H, W) for DeepFool
                pert_input = pert_image.squeeze(0)

                try:
                    dr, num_iter, _, final_label = deepfool(
                        pert_input,
                        model,
                        num_classes=num_classes,
                        overshoot=overshoot,
                        max_iter=max_iter_df
                    )

                    # Validate DeepFool output
                    if (dr is not None and
                        torch.is_tensor(dr) and
                        dr.numel() > 0 and
                        torch.norm(dr.flatten()) > 1e-8):

                        # Ensure dr has the correct shape and device
                        dr = dr.to(device)
                        if dr.shape != v.shape:
                            print(f"Warning: Shape mismatch - dr: {dr.shape}, v: {v.shape}")
                            continue

                        # Check if DeepFool actually fooled the image
                        if final_label != pert_pred:
                            # Update universal perturbation
                            old_v_norm = torch.norm(v.flatten(), p=2).item()
                            v = v + dr

                            # Project to constraint set
                            v = project_perturbation(v, xi, norm_type)
                            new_v_norm = torch.norm(v.flatten(), p=2).item()

                            print(f"  Updated UAP: {old_v_norm:.6f} -> {new_v_norm:.6f}")

                            # Re-check if now fooled
                            with torch.no_grad():
                                new_pert_image = image + v
                                new_pred = model(new_pert_image).argmax(dim=1).item()
                                if new_pred != orig_pred:
                                    fooled_count += 1
                        else:
                            print(f"  DeepFool failed to fool image")
                    else:
                        print(f"  DeepFool returned invalid perturbation")

                except Exception as e:
                    print(f"  DeepFool error: {e}")
                    continue
            else:
                fooled_count += 1

            total_count += 1

        # Compute fooling rate on the training subset
        fooling_rate = fooled_count / total_count

        # Compute fooling rate on full dataset for monitoring
        # (Optional: can be expensive, comment out if too slow)
        # full_fooling_rate = compute_fooling_rate(model, dataset, v, device)

        fooling_rates.append(fooling_rate)

        print(f"Fooling rate: {fooling_rate:.2%} ({fooled_count}/{total_count})")
        print(f"Perturbation L2 norm: {torch.norm(v.flatten(), p=2).item():.4f}")
        print(f"Perturbation L∞ norm: {torch.max(torch.abs(v)).item():.4f}")

        # Save checkpoint
        if save_path and (iteration + 1) % save_interval == 0:
            # Handle both file paths and directory paths
            if save_path.endswith('.pt'):
                # It's a file path
                checkpoint_path = save_path.replace('.pt', f'_iter{iteration + 1}.pt')
            else:
                # It's a directory path
                checkpoint_path = os.path.join(save_path, f'uap_perturbation_iter{iteration + 1}.pt')

            torch.save({
                'perturbation': v.cpu(),
                'iteration': iteration + 1,
                'fooling_rate': fooling_rate,
                'fooling_rates': fooling_rates,
                'config': {
                    'xi': xi,
                    'delta': delta,
                    'norm_type': norm_type,
                    'max_iter_uni': max_iter_uni,
                }
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")

        # Early stopping if target fooling rate reached
        if fooling_rate >= delta:
            print(f"\n✓ Target fooling rate {delta:.2%} reached!")
            break

    # Save final perturbation
    if save_path:
        # Handle both file paths and directory paths
        if save_path.endswith('.pt'):
            # It's a file path
            final_path = save_path
        else:
            # It's a directory path
            final_path = os.path.join(save_path, 'uap_final.pt')

        torch.save({
            'perturbation': v.cpu(),
            'iteration': iteration + 1,
            'fooling_rate': fooling_rate,
            'fooling_rates': fooling_rates,
            'config': {
                'xi': xi,
                'delta': delta,
                'norm_type': norm_type,
                'max_iter_uni': max_iter_uni,
            }
        }, final_path)
        print(f"\n✓ Final perturbation saved: {final_path}")

    print(f"\n{'='*80}")
    print(f"UAP Computation Complete")
    print(f"{'='*80}")
    print(f"Final fooling rate: {fooling_rate:.2%}")
    print(f"Iterations completed: {iteration + 1}")
    print(f"{'='*80}\n")

    return v, fooling_rates


def load_uap(path: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Load a saved universal perturbation

    Args:
        path: Path to saved perturbation
        device: Device to load on

    Returns:
        Universal perturbation tensor
    """
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        perturbation = checkpoint['perturbation']
        print(f"✓ Loaded UAP from {path}")
        if 'fooling_rate' in checkpoint:
            print(f"  Fooling rate: {checkpoint['fooling_rate']:.2%}")
        if 'iteration' in checkpoint:
            print(f"  Iteration: {checkpoint['iteration']}")
    else:
        perturbation = checkpoint

    return perturbation.to(device)


if __name__ == "__main__":
    """Test UAP implementation"""
    print("Testing UAP implementation...")

    from torchvision import models, datasets, transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create a simple model for testing
    model = models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)
    model.eval()

    # Create a small test dataset (CIFAR-10 for quick testing)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Use a very small subset for testing
    full_dataset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
    subset_indices = list(range(100))  # Only 100 images for testing
    test_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

    print(f"\nTest dataset size: {len(test_dataset)}")

    # Compute UAP with small parameters for testing
    print("\nComputing UAP (this may take a few minutes)...")
    v, fooling_rates = compute_uap(
        model=model,
        dataset=test_dataset,
        num_classes=10,
        xi=10/255,
        delta=0.5,  # Lower target for testing
        max_iter_uni=2,  # Only 2 iterations for testing
        norm_type='inf',
        max_iter_df=10,  # Fewer DeepFool iterations
        device=device
    )

    print(f"\nFinal UAP shape: {v.shape}")
    print(f"Fooling rates: {fooling_rates}")

    print("\n✓ UAP test completed!")
