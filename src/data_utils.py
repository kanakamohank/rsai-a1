"""
Dataset utilities for CIFAR-100
"""
import torch
import torchvision
from torchvision import datasets, transforms
from typing import Tuple, Optional
import random


def get_cifar100_loaders(data_root: str = './data',
                         batch_size: int = 128,
                         num_workers: int = 4) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get CIFAR-100 train and test data loaders

    Args:
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of worker threads

    Returns:
        train_loader, test_loader
    """
    # CIFAR-100 normalization parameters
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_cifar100_subset(data_root: str = './data',
                       subset_size: int = 1000,
                       train: bool = True,
                       seed: Optional[int] = None) -> torch.utils.data.Dataset:
    """
    Get a random subset of CIFAR-100 for UAP computation

    Args:
        data_root: Root directory for data
        subset_size: Number of images in subset
        train: Whether to use training set
        seed: Random seed for reproducibility

    Returns:
        Subset dataset
    """
    # Normalization for CIFAR-100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load full dataset
    full_dataset = datasets.CIFAR100(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Sample random subset
    total_size = len(full_dataset)
    subset_size = min(subset_size, total_size)
    indices = random.sample(range(total_size), subset_size)

    # Create subset
    subset = torch.utils.data.Subset(full_dataset, indices)

    print(f"✓ Created CIFAR-100 subset: {len(subset)} images")
    return subset


def get_unnormalized_dataset(data_root: str = './data',
                             train: bool = False) -> torch.utils.data.Dataset:
    """
    Get CIFAR-100 dataset without normalization (for visualization)

    Args:
        data_root: Root directory for data
        train: Whether to use training set

    Returns:
        Dataset with only ToTensor transform
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR100(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )

    return dataset


def denormalize_cifar100(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize CIFAR-100 tensor for visualization

    Args:
        tensor: Normalized tensor

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)

    denormalized = tensor * std + mean
    return denormalized


if __name__ == "__main__":
    """Test data utilities"""
    print("Testing data utilities...")

    # Test data loaders
    print("\n1. Testing data loaders...")
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Get a sample batch
    images, labels = next(iter(test_loader))
    print(f"   Sample batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Test subset creation
    print("\n2. Testing subset creation...")
    subset = get_cifar100_subset(subset_size=100, seed=42)
    print(f"   Subset size: {len(subset)}")

    sample_image, sample_label = subset[0]
    print(f"   Sample image shape: {sample_image.shape}")
    print(f"   Sample label: {sample_label}")

    # Test denormalization
    print("\n3. Testing denormalization...")
    denorm_image = denormalize_cifar100(sample_image)
    print(f"   Denormalized image range: [{denorm_image.min():.3f}, {denorm_image.max():.3f}]")

    print("\n✓ All data utility tests passed!")
