"""
Projected Gradient Descent (PGD) Attack

Reference: Madry et al. (2018)
"Towards Deep Learning Models Resistant to Adversarial Attacks"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def pgd_attack(model: nn.Module,
               image: torch.Tensor,
               label: torch.Tensor,
               epsilon: float = 10/255,
               alpha: float = 2/255,
               num_iter: int = 10,
               random_start: bool = True,
               targeted: bool = False,
               target_label: Optional[int] = None) -> Tuple[torch.Tensor, bool]:
    """
    Projected Gradient Descent (PGD) attack

    PGD is an iterative version of FGSM that takes multiple smaller steps
    and projects back to the epsilon ball after each step.

    Args:
        model: Neural network model
        image: Input image tensor of shape (1, C, H, W) or (C, H, W)
        label: True label (for untargeted) or target label (for targeted)
        epsilon: Maximum perturbation magnitude (L∞ bound)
        alpha: Step size for each iteration
        num_iter: Number of iterations
        random_start: Whether to start from random point in epsilon ball
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

    # Initialize perturbation
    if random_start:
        # Start from random point in epsilon ball
        perturbation = torch.empty_like(image).uniform_(-epsilon, epsilon)
        perturbed_image = torch.clamp(image + perturbation, 0, 1)
    else:
        # Start from original image
        perturbed_image = image.clone()

    # PGD iterations
    for i in range(num_iter):
        # Require gradient
        perturbed_image = perturbed_image.detach().clone().requires_grad_(True)

        # Forward pass
        output = model(perturbed_image)

        # Compute loss
        if targeted:
            # For targeted attack, minimize loss for target class
            if target_label is None:
                raise ValueError("target_label must be provided for targeted attack")
            target = torch.tensor([target_label]).to(device)
            loss = -F.cross_entropy(output, target)
        else:
            # For untargeted attack, maximize loss for true class
            loss = F.cross_entropy(output, label)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient
        data_grad = perturbed_image.grad.data

        # Take step in direction of gradient (for untargeted)
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * data_grad.sign()

            # Project back to epsilon ball around original image
            perturbation = perturbed_image - image
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
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


def pgd_batch(model: nn.Module,
              images: torch.Tensor,
              labels: torch.Tensor,
              epsilon: float = 10/255,
              alpha: float = 2/255,
              num_iter: int = 10,
              random_start: bool = True) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply PGD attack to a batch of images

    Args:
        model: Neural network model
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        num_iter: Number of iterations
        random_start: Whether to start from random point

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

    # Initialize perturbation
    if random_start:
        perturbation = torch.empty_like(images).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(images + perturbation, 0, 1)
    else:
        perturbed_images = images.clone()

    # PGD iterations
    for i in range(num_iter):
        # Require gradient
        perturbed_images = perturbed_images.detach().clone().requires_grad_(True)

        # Forward pass
        outputs = model(perturbed_images)

        # Compute loss
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient
        data_grad = perturbed_images.grad.data

        # Take step
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * data_grad.sign()

            # Project back to epsilon ball
            perturbation = perturbed_images - images
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            perturbed_images = images + perturbation

            # Clamp to valid range
            perturbed_images = torch.clamp(perturbed_images, 0, 1)

    # Get final predictions
    with torch.no_grad():
        final_outputs = model(perturbed_images)
        final_preds = final_outputs.argmax(dim=1)

    # Calculate success rate
    success = (final_preds != orig_preds).float()
    success_rate = success.mean().item()

    return perturbed_images, final_preds, success_rate


if __name__ == "__main__":
    """Test PGD implementation"""
    print("Testing PGD implementation...")

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

    print("\n1. Testing untargeted PGD...")
    print(f"   Input shape: {test_image.shape}")

    # Get original prediction
    with torch.no_grad():
        orig_output = model(test_image)
        orig_pred = orig_output.argmax(dim=1).item()
        orig_conf = torch.softmax(orig_output, dim=1).max().item()

    print(f"   Original prediction: {orig_pred} (confidence: {orig_conf:.4f})")

    # Apply PGD
    perturbed_image, success = pgd_attack(
        model, test_image, test_label,
        epsilon=10/255, alpha=2/255, num_iter=10
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
    print(f"   Maximum allowed L∞: {10/255:.6f}")

    # Test without random start
    print("\n2. Testing PGD without random start...")
    perturbed_image2, success2 = pgd_attack(
        model, test_image, test_label,
        epsilon=10/255, alpha=2/255, num_iter=10, random_start=False
    )
    print(f"   Attack success: {success2}")

    # Test batch attack
    print("\n3. Testing batch PGD...")
    batch_images = torch.rand(10, 3, 32, 32).to(device)
    batch_labels = torch.randint(0, 100, (10,)).to(device)

    perturbed_batch, preds, success_rate = pgd_batch(
        model, batch_images, batch_labels,
        epsilon=10/255, alpha=2/255, num_iter=10
    )

    print(f"   Batch size: {batch_images.shape[0]}")
    print(f"   Success rate: {success_rate:.2%}")
    print(f"   Perturbed batch shape: {perturbed_batch.shape}")

    # Compare PGD with different iterations
    print("\n4. Testing PGD with different iterations...")
    for iters in [1, 5, 10, 20]:
        _, success = pgd_attack(
            model, test_image, test_label,
            epsilon=10/255, alpha=2/255, num_iter=iters
        )
        print(f"   {iters} iterations: success = {success}")

    print("\n✓ PGD test completed!")
