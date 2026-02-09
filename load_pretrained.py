"""
Load pre-trained CIFAR-100 model from torch.hub
"""
import torch
import os
from src.data_utils import get_cifar100_loaders
from config import DEVICE, PATHS

def load_pretrained_model(model_name='cifar100_resnet20', device=None):
    """
    Load pre-trained CIFAR-100 model from torch.hub

    Args:
        model_name: Name of the model (e.g., 'cifar100_resnet20', 'cifar100_vgg16_bn')
        device: Device to load model on

    Returns:
        model: Pre-trained model
    """
    if device is None:
        device = DEVICE

    print(f"Loading pre-trained model: {model_name}")
    print(f"Device: {device}")

    # Load from torch.hub
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
        trust_repo=True
    )

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def test_model_accuracy(model, device=None):
    """Test model accuracy on CIFAR-100 test set"""
    if device is None:
        device = DEVICE

    print("\nTesting model accuracy on CIFAR-100 test set...")

    # Load test data
    _, test_loader = get_cifar100_loaders(
        data_root=PATHS['data'],
        batch_size=128,
        num_workers=0  # Disable multiprocessing for macOS MPS compatibility
    )

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
    print(f"✓ Test Accuracy: {accuracy:.2f}%")

    return accuracy


def save_model(model, model_name, save_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
    }

    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load pre-trained CIFAR-100 model')
    parser.add_argument('--model', type=str, default='cifar100_resnet20',
                       choices=['cifar100_resnet20', 'cifar100_resnet32', 'cifar100_resnet44',
                               'cifar100_vgg16_bn', 'cifar100_vgg19_bn'],
                       help='Model to load')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save model (default: checkpoints/{model_name}.pth)')

    args = parser.parse_args()

    # Load model
    model = load_pretrained_model(args.model, DEVICE)

    # Test accuracy
    accuracy = test_model_accuracy(model, DEVICE)

    # Save model
    if args.save_path is None:
        args.save_path = f"{PATHS['checkpoints']}/{args.model}.pth"

    save_model(model, args.model, args.save_path)

    print(f"\n{'='*60}")
    print(f"✓ Pre-trained model ready!")
    print(f"  Model: {args.model}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Saved to: {args.save_path}")
    print(f"{'='*60}")
    print(f"\nUpdate config.py with:")
    print(f"  MODEL_PATH = '{args.save_path}'")
