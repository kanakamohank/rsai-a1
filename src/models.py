"""
Model loading and preparation utilities
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def get_model(model_name: str, num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """
    Load a model architecture

    Args:
        model_name: Name of the model ('resnet18', 'vgg16', etc.)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (ImageNet)

    Returns:
        PyTorch model
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        # Modify final layer for CIFAR-100
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        # Modify classifier for CIFAR-100
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def load_cifar100_model(model_name: str, checkpoint_path: Optional[str] = None,
                        device: torch.device = torch.device('cpu')) -> nn.Module:
    """
    Load a pre-trained CIFAR-100 model

    Supports both:
    - Custom models (resnet18, vgg16) from get_model()
    - Torch.hub models (cifar100_resnet32, cifar100_vgg16_bn, etc.)

    Args:
        model_name: Name of the model architecture
        checkpoint_path: Path to saved model weights
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Check if it's a torch.hub model (starts with 'cifar')
    if model_name.startswith('cifar100_') or model_name.startswith('cifar10_'):
        # Load from torch.hub
        print(f"Loading torch.hub model: {model_name}")
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            model_name,
            pretrained=False,  # We'll load our saved checkpoint
            trust_repo=True,
            verbose=False
        )

        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

                print(f"✓ Loaded weights from {checkpoint_path}")
            except FileNotFoundError:
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
                print("  Using randomly initialized model")
    else:
        # Use custom get_model() for standard architectures
        model = get_model(model_name, num_classes=100, pretrained=False)

        if checkpoint_path is not None:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

                print(f"✓ Loaded model from {checkpoint_path}")
            except FileNotFoundError:
                print(f"⚠ Checkpoint not found: {checkpoint_path}")
                print("  Using randomly initialized model")
        else:
            print("⚠ No checkpoint provided, using randomly initialized model")

    model = model.to(device)
    model.eval()  # Set to evaluation mode

    return model


def test_model_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                       device: torch.device) -> float:
    """
    Test model accuracy on a dataset

    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run on

    Returns:
        Accuracy (0-100)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def save_model(model: nn.Module, path: str, epoch: Optional[int] = None,
               optimizer: Optional[torch.optim.Optimizer] = None,
               accuracy: Optional[float] = None):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        path: Save path
        epoch: Current epoch (optional)
        optimizer: Optimizer state (optional)
        accuracy: Model accuracy (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy

    torch.save(checkpoint, path)
    print(f"✓ Model saved to {path}")


if __name__ == "__main__":
    # Test model loading
    print("Testing model utilities...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test ResNet-18
    print("\n1. Testing ResNet-18...")
    model = get_model('resnet18', num_classes=100)
    model.eval()  # Set to eval mode for testing
    print(f"   Model created: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    assert output.shape == (1, 100), "Output shape mismatch"

    # Test VGG-16
    print("\n2. Testing VGG-16...")
    model = get_model('vgg16', num_classes=100)
    model.eval()  # Set to eval mode for testing
    print(f"   Model created: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✓ All tests passed!")
