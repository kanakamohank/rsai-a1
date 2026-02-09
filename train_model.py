"""
Train a model on CIFAR-100 from scratch

This script trains a ResNet-18 or VGG-16 model on CIFAR-100.
You can skip this if you already have a pre-trained model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import os
from tqdm import tqdm

from config import *
from src.models import get_model, save_model, test_model_accuracy
from src.data_utils import get_cifar100_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.3f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(model_name='resnet18', num_epochs=100, batch_size=128,
                learning_rate=0.1, device=None):
    """
    Train a model on CIFAR-100

    Args:
        model_name: Model architecture ('resnet18' or 'vgg16')
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()} on CIFAR-100")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading CIFAR-100 dataset...")
    train_loader, test_loader = get_cifar100_loaders(
        data_root=PATHS['data'],
        batch_size=batch_size
    )
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}\n")

    # Create model
    print(f"Creating {model_name} model...")
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model = model.to(device)
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=TRAIN_CONFIG['momentum'],
                         weight_decay=TRAIN_CONFIG['weight_decay'])

    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer,
                           milestones=TRAIN_CONFIG['scheduler_milestones'],
                           gamma=TRAIN_CONFIG['scheduler_gamma'])

    # Create checkpoints directory
    os.makedirs(PATHS['checkpoints'], exist_ok=True)

    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, device)

        # Evaluate
        test_acc = test_model_accuracy(model, test_loader, device)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(PATHS['checkpoints'],
                                    f'{model_name}_cifar100_best.pth')
            save_model(model, save_path, epoch=epoch + 1,
                      optimizer=optimizer, accuracy=test_acc)
            print(f"✓ New best accuracy: {best_acc:.2f}%")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(PATHS['checkpoints'],
                                    f'{model_name}_cifar100_epoch{epoch + 1}.pth')
            save_model(model, save_path, epoch=epoch + 1,
                      optimizer=optimizer, accuracy=test_acc)

    # Save final model
    final_path = os.path.join(PATHS['checkpoints'],
                             f'{model_name}_cifar100_final.pth')
    save_model(model, final_path, epoch=num_epochs,
              optimizer=optimizer, accuracy=test_acc)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(PATHS['checkpoints'], f'{model_name}_cifar100_best.pth')}")
    print(f"{'='*80}\n")

    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train model on CIFAR-100')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'vgg16'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Train model
    train_model(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
