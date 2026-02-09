"""
Configuration file for UAP project
"""
import torch

# Device configuration
# Use MPS (Metal Performance Shaders) for Apple Silicon GPUs
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Dataset configuration
DATASET = 'CIFAR100'
NUM_CLASSES = 100
IMAGE_SIZE = 32
NUM_CHANNELS = 3

# CIFAR-100 normalization parameters
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

# Model configuration - Using deeper pretrained models for better performance
MODEL_NAME = 'cifar100_resnet32'  # Upgraded from ResNet-20 (70.14% accuracy)
MODEL_PATH = 'checkpoints/cifar100_resnet32.pth'

# Alternative models for transferability testing
MODEL_NAME_TRANSFER = 'cifar100_vgg19_bn'  # Upgraded from VGG-16 (73.85% accuracy)
MODEL_PATH_TRANSFER = 'checkpoints/cifar100_vgg19_bn.pth'

# UAP configuration - Enhanced for better correlation and transferability
UAP_CONFIG = {
    'subset_size': 700,         # Increased for better gradient statistics
    'xi': 20/255,               # Reduced perturbation for better transferability
    'delta': 0.65,              # Slightly lower target to allow more iterations
    'max_iter_uni': 3,          # More iterations for better convergence
    'norm_type': 'inf',         # 'inf' or 2
    'save_interval': 1,         # Save checkpoint every N iterations
}

# DeepFool configuration
DEEPFOOL_CONFIG = {
    'max_iter': 20,             # Maximum iterations
    'overshoot': 0.02,          # Overshoot parameter
}

# FGSM configuration
FGSM_CONFIG = {
    'epsilon': 10/255,          # Perturbation budget
}

# PGD configuration
PGD_CONFIG = {
    'epsilon': 10/255,          # Perturbation budget
    'alpha': 2/255,             # Step size
    'num_iter': 10,             # Number of iterations
}

# Analysis configuration - Enhanced for better statistics
ANALYSIS_CONFIG = {
    'num_gradient_samples': 200,    # Doubled for better correlation estimation
    'num_pca_samples': 800,         # Increased for better dimensionality analysis
    'num_transfer_samples': 1500,   # More samples for robust transferability testing
}

# Paths
PATHS = {
    'data': './data',
    'checkpoints': './checkpoints',
    'results': './results',
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Training configuration (if training model from scratch)
TRAIN_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_epochs': 100,
    'scheduler_milestones': [60, 80],
    'scheduler_gamma': 0.1,
}
