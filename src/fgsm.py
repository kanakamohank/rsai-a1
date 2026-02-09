"""
Fast Gradient Sign Method (FGSM) Attack

Reference: Goodfellow et al. (2015)
"Explaining and Harnessing Adversarial Examples"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def fgsm_attack(model: nn.Module,
                image: torch.Tensor,
                label: torch.Tensor,
                epsilon: float = 10/255,
                targeted: bool = False,
                target_label: int = None) -> Tuple[torch.Tensor, bool]:
    """
    Fast Gradient Sign Method (FGSM) attack

    Args:
        model: Neural network model
        image: Input image tensor of shape (1, C, H, W) or (C, H, W)
        label: True label (for untargeted) or target label (for targeted)
        epsilon: Perturbation magnitude (L∞ bound)
        targeted: Whether to perform targeted attack
        target_label: Target class for targeted attack

    Returns:
        perturbed_image: Adversarial example
        success: Whether attack succeeded
    """
    # Ensure image has batch dimension
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # Ensure label is tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor([label])

    # Move to same device as model
    device = next(model.parameters()).device
    image = image.to(device)
    label = label.to(device)

    # Set model to eval mode
    model.eval()

    # Get original prediction
    with torch.no_grad():
        orig_output = model(image)
        orig_pred = orig_output.argmax(dim=1).item()

    # Require gradient for input
    image_adv = image.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(image_adv)

    # Compute loss
    if targeted:
        # For targeted attack, minimize loss for target class
        if target_label is None:
            raise ValueError("target_label must be provided for targeted attack")
        target = torch.tensor([target_label]).to(device)
        loss = -F.cross_entropy(output, target)  # Negative to maximize target class
    else:
        # For untargeted attack, maximize loss for true class
        loss = F.cross_entropy(output, label)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get gradient sign
    data_grad = image_adv.grad.data

    # Create perturbation
    perturbation = epsilon * data_grad.sign()

    # Apply perturbation
    perturbed_image = image + perturbation

    # Clamp to valid range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Check if attack succeeded
    with torch.no_grad():
        final_output = model(perturbed_image)
        final_pred = final_output.argmax(dim=1).item()

    if targeted:
        success = (final_pred == target_label)
    else:
        success = (final_pred != orig_pred)

    # Remove batch dimension if input was single image
    perturbed_image = perturbed_image.squeeze(0)

    return perturbed_image, success


def fgsm_batch(model: nn.Module,
               images: torch.Tensor,
               labels: torch.Tensor,
               epsilon: float = 10/255) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply FGSM attack to a batch of images

    Args:
        model: Neural network model
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        epsilon: Perturbation magnitude

    Returns:
        perturbed_images: Batch of adversarial examples
        predictions: Predictions on adversarial examples
        success_rate: Percentage of successful attacks
    """
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)

    model.eval()

    # Get original predictions
    with torch.no_grad():
        orig_outputs = model(images)
        orig_preds = orig_outputs.argmax(dim=1)

    # Require gradient
    images_adv = images.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(images_adv)

    # Compute loss
    loss = F.cross_entropy(outputs, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Get gradient sign
    data_grad = images_adv.grad.data

    # Create perturbations
    perturbations = epsilon * data_grad.sign()

    # Apply perturbations
    perturbed_images = images + perturbations

    # Clamp to valid range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Get final predictions
    with torch.no_grad():
        final_outputs = model(perturbed_images)
        final_preds = final_outputs.argmax(dim=1)

    # Calculate success rate (predictions changed)
    success = (final_preds != orig_preds).float()
    success_rate = success.mean().item()

    return perturbed_images, final_preds, success_rate


if __name__ == "__main__":
    """Test FGSM implementation"""
    print("Testing FGSM implementation...")

    from torchvision import models
    import torch

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create a simple model for testing
    model = models.resnet18(pretrained=False, num_classes=100)
    model = model.to(device)
    model.eval()

    # Create random test image and label
    test_image = torch.rand(1, 3, 32, 32).to(device)
    test_label = torch.tensor([5]).to(device)

    print("\n1. Testing untargeted FGSM...")
    print(f"   Input shape: {test_image.shape}")

    # Get original prediction
    with torch.no_grad():
        orig_output = model(test_image)
        orig_pred = orig_output.argmax(dim=1).item()
        orig_conf = torch.softmax(orig_output, dim=1).max().item()

    print(f"   Original prediction: {orig_pred} (confidence: {orig_conf:.4f})")

    # Apply FGSM
    perturbed_image, success = fgsm_attack(
        model, test_image, test_label, epsilon=10/255
    )

    print(f"   Perturbation shape: {perturbed_image.shape}")

    # Get adversarial prediction
    with torch.no_grad():
        adv_output = model(perturbed_image.unsqueeze(0))
        adv_pred = adv_output.argmax(dim=1).item()
        adv_conf = torch.softmax(adv_output, dim=1).max().item()

    print(f"   Adversarial prediction: {adv_pred} (confidence: {adv_conf:.4f})")
    print(f"   Attack success: {success}")

    # Compute perturbation statistics
    perturbation = perturbed_image - test_image.squeeze(0)
    l2_norm = torch.norm(perturbation).item()
    linf_norm = torch.max(torch.abs(perturbation)).item()

    print(f"   Perturbation L2 norm: {l2_norm:.6f}")
    print(f"   Perturbation L∞ norm: {linf_norm:.6f}")
    print(f"   Expected L∞: {10/255:.6f}")

    # Test batch attack
    print("\n2. Testing batch FGSM...")
    batch_images = torch.rand(10, 3, 32, 32).to(device)
    batch_labels = torch.randint(0, 100, (10,)).to(device)

    perturbed_batch, preds, success_rate = fgsm_batch(
        model, batch_images, batch_labels, epsilon=10/255
    )

    print(f"   Batch size: {batch_images.shape[0]}")
    print(f"   Success rate: {success_rate:.2%}")
    print(f"   Perturbed batch shape: {perturbed_batch.shape}")

    print("\n✓ FGSM test completed!")
