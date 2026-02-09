"""
DeepFool algorithm implementation

Reference: Moosavi-Dezfooli et al. (2016)
"DeepFool: a simple and accurate method to fool deep neural networks"
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def deepfool(image: torch.Tensor,
             model: nn.Module,
             num_classes: int = 100,
             overshoot: float = 0.02,
             max_iter: int = 25) -> Tuple[torch.Tensor, int, torch.Tensor, str]:
    """
    DeepFool algorithm to compute minimal adversarial perturbation

    Args:
        image: Input image tensor of shape (1, C, H, W) or (C, H, W)
        model: Neural network model
        num_classes: Number of classes
        overshoot: Overshoot parameter for crossing boundary
        max_iter: Maximum number of iterations

    Returns:
        perturbation: Adversarial perturbation (same shape as image)
        num_iter: Number of iterations used
        pert_image: Perturbed image
        pred_label: Final predicted label
    """
    # Ensure image has batch dimension
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # Clone image and enable gradients
    image = image.clone().detach()
    image.requires_grad = True

    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)

    # Get original prediction
    with torch.no_grad():
        output = model(image)
        original_label = output.argmax(dim=1).item()

    # Initialize
    pert_image = image.clone()
    total_perturbation = torch.zeros_like(image)

    for iteration in range(max_iter):
        # Clone and detach to make it a leaf tensor
        pert_image = pert_image.detach().clone().requires_grad_(True)

        # Forward pass
        output = model(pert_image)

        # Current prediction
        current_label = output.argmax(dim=1).item()

        # If already fooled, return
        if current_label != original_label:
            break

        # Get scores for all classes
        scores = output[0]

        # Sort classes by score (descending)
        sorted_classes = scores.argsort(descending=True)
        k_0 = sorted_classes[0].item()  # Current predicted class

        # Find the closest decision boundary
        min_distance = float('inf')
        min_perturbation = None

        # Iterate over all other classes
        for k in range(1, num_classes):
            k_i = sorted_classes[k].item()

            # Zero gradients
            if pert_image.grad is not None:
                pert_image.grad.zero_()

            # Compute gradient of f_{k_0}
            scores[k_0].backward(retain_graph=True)
            grad_k0 = pert_image.grad.clone()

            # Zero gradients
            pert_image.grad.zero_()

            # Compute gradient of f_{k_i}
            scores[k_i].backward(retain_graph=True)
            grad_ki = pert_image.grad.clone()

            # Compute w_k = grad(f_{k_0}) - grad(f_{k_i})
            w_k = grad_k0 - grad_ki

            # Compute f_k = f_{k_0} - f_{k_i}
            f_k = (scores[k_0] - scores[k_i]).detach()

            # Compute distance to boundary
            # distance = |f_k| / ||w_k||_2^2
            w_k_norm_sq = torch.sum(w_k ** 2)

            if w_k_norm_sq < 1e-10:  # Numerical stability
                continue

            distance = torch.abs(f_k) / w_k_norm_sq

            # Keep track of minimum distance
            if distance < min_distance:
                min_distance = distance
                min_perturbation = (torch.abs(f_k) / w_k_norm_sq) * w_k

        # If no valid perturbation found, break
        if min_perturbation is None:
            break

        # Update perturbation with overshoot
        r_i = (1 + overshoot) * min_perturbation
        total_perturbation += r_i.detach()

        # Update perturbed image
        pert_image = (image + total_perturbation).detach()

        # Ensure valid range [0, 1] if images are normalized to [0, 1]
        # Note: If using ImageNet normalization, this should be adjusted
        pert_image = torch.clamp(pert_image, 0, 1)

    # Get final prediction
    with torch.no_grad():
        final_output = model(pert_image)
        final_label = final_output.argmax(dim=1).item()

    # Remove batch dimension if input was single image
    perturbation = total_perturbation.squeeze(0)
    pert_image = pert_image.squeeze(0)

    return perturbation, iteration + 1, pert_image, final_label


def deepfool_batch(images: torch.Tensor,
                   model: nn.Module,
                   num_classes: int = 100,
                   overshoot: float = 0.02,
                   max_iter: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply DeepFool to a batch of images (processed sequentially)

    Args:
        images: Batch of images (B, C, H, W)
        model: Neural network model
        num_classes: Number of classes
        overshoot: Overshoot parameter
        max_iter: Maximum iterations

    Returns:
        perturbations: Batch of perturbations (B, C, H, W)
        pert_images: Batch of perturbed images (B, C, H, W)
    """
    batch_size = images.size(0)
    perturbations = []
    pert_images = []

    for i in range(batch_size):
        image = images[i:i+1]
        pert, _, pert_img, _ = deepfool(image, model, num_classes, overshoot, max_iter)
        perturbations.append(pert.unsqueeze(0))
        pert_images.append(pert_img.unsqueeze(0))

    perturbations = torch.cat(perturbations, dim=0)
    pert_images = torch.cat(pert_images, dim=0)

    return perturbations, pert_images


if __name__ == "__main__":
    """Test DeepFool implementation"""
    print("Testing DeepFool implementation...")

    # Create a simple test model
    from torchvision import models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load a simple model (random weights for testing)
    model = models.resnet18(pretrained=False, num_classes=100)
    model = model.to(device)
    model.eval()

    # Create a random test image
    test_image = torch.rand(1, 3, 32, 32).to(device)

    print("\nRunning DeepFool...")
    print(f"Input image shape: {test_image.shape}")

    # Get original prediction
    with torch.no_grad():
        orig_output = model(test_image)
        orig_label = orig_output.argmax(dim=1).item()
        orig_confidence = torch.softmax(orig_output, dim=1).max().item()

    print(f"Original prediction: {orig_label} (confidence: {orig_confidence:.4f})")

    # Apply DeepFool
    perturbation, num_iter, pert_image, final_label = deepfool(
        test_image, model, num_classes=100, max_iter=50
    )

    print(f"\nDeepFool completed in {num_iter} iterations")
    print(f"Final prediction: {final_label}")
    print(f"Perturbation shape: {perturbation.shape}")
    print(f"Perturbation L2 norm: {torch.norm(perturbation).item():.6f}")
    print(f"Perturbation L∞ norm: {torch.max(torch.abs(perturbation)).item():.6f}")

    # Check if attack was successful
    if final_label != orig_label:
        print("✓ Attack successful: Prediction changed")
    else:
        print("✗ Attack failed: Prediction unchanged")

    print("\n✓ DeepFool test completed!")
