# Universal Adversarial Perturbations (UAP) - CIFAR-100

Implementation of Universal Adversarial Perturbations following Moosavi-Dezfooli et al. (CVPR 2017) for CIFAR-100 dataset.

## Overview

This project implements and analyzes Universal Adversarial Perturbations (UAPs) - single perturbation vectors that can fool deep neural networks across multiple images, unlike traditional per-image adversarial attacks.

**What is a UAP?** A single perturbation vector `v` that satisfies `f(x + v) ≠ f(x)` for most images `x` from a distribution, subject to a norm constraint `||v|| ≤ ξ`. Unlike FGSM or PGD which compute a different perturbation for each image, UAP finds one "master key" that works universally.

## Project Structure

```
rsai-a1/
├── README.md                  # This file - project overview and navigation
├── IMPLEMENTATION_PLAN.md     # ⭐ Comprehensive guide with all details
├── QUICK_REFERENCE.md         # ⭐ Quick checklist and commands
├── COLAB_GUIDE.md             # ⭐ Google Colab execution guide
├── UAP_Full_Experiment.ipynb  # 🎯 Colab notebook (complete pipeline)
├── config.py                  # Configuration and hyperparameters
├── requirements.txt           # Python dependencies
├── main.py                    # Main execution script
├── load_pretrained.py         # Load pre-trained models from torch.hub
├── train_model.py             # Train CIFAR-100 model (optional)
├── compare_attacks.py         # ✓ Compare UAP, FGSM, PGD attacks
├── analyze_uap.py             # ✓ Comprehensive analysis script
├── run_full_pipeline.py       # ✓ Run complete pipeline (UAP + Analysis)
├── src/                       # Source code
│   ├── __init__.py
│   ├── models.py              # ✓ Model loading and utilities
│   ├── data_utils.py          # ✓ CIFAR-100 dataset utilities
│   ├── deepfool.py            # ✓ DeepFool algorithm
│   ├── uap.py                 # ✓ UAP algorithm
│   ├── fgsm.py                # ✓ FGSM baseline
│   ├── pgd.py                 # ✓ PGD baseline
│   ├── evaluation.py          # ✓ Evaluation and visualization
│   └── analysis.py            # ✓ Analysis experiments
├── data/                      # CIFAR-100 dataset (auto-downloaded)
├── checkpoints/               # Pre-trained models
│   ├── cifar100_resnet32.pth  # ✓ ResNet-32 (70.14% accuracy)
│   └── cifar100_vgg16_bn.pth  # ✓ VGG-16 (74.00% accuracy)
└── results/                   # Generated perturbations and figures
```

## Key Features

- **DeepFool Algorithm**: Minimal perturbation computation
- **UAP Algorithm**: Universal perturbation generation
- **Baseline Attacks**: FGSM and PGD implementations
- **Comprehensive Analysis**:
  - Gradient correlation analysis
  - Dimensionality analysis (PCA)
  - Transferability testing
- **Visualization**: Perturbations, convergence, and comparative analysis

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- ~2GB disk space

## Getting Started

### 📚 Documentation Guide

Before starting, familiarize yourself with the documentation:

1. **Start here:** Read this README for project overview
2. **For detailed implementation:** See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
   - Complete algorithm explanations (DeepFool + UAP)
   - Phase-by-phase implementation guide
   - Analysis experiment templates
   - Troubleshooting guide
3. **For quick reference:** See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
   - Assignment requirements checklist
   - Quick command reference
   - Expected results table
   - Common pitfalls

### ⚡ Quick Start

**Option 1: Google Colab (Recommended - Free GPU!)**
```
1. Upload UAP_Full_Experiment.ipynb to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells (Runtime → Run all)
4. Wait 1-2 hours
5. Download results ZIP

See COLAB_GUIDE.md for detailed instructions
```

**Option 2: Local Execution**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load pre-trained models (recommended)
python load_pretrained.py --model cifar100_resnet32
python load_pretrained.py --model cifar100_vgg16_bn

# 3. Verify model accuracy (should be >70%)
python main.py --test-model

# 4. Run full pipeline (UAP + Analysis)
python run_full_pipeline.py

# OR run steps individually:

# 4a. Compute UAP (~3-4 hours on MPS/GPU)
python main.py --compute-uap

# 4b. Run comprehensive analysis
python analyze_uap.py --uap-path ./results/uap_final.pt --output-dir ./results/analysis

# 4c. Compare with baseline attacks
python compare_attacks.py --uap-path ./results/uap_final.pt
```

### 📖 Detailed Instructions

For step-by-step instructions with explanations, see:
- **IMPLEMENTATION_PLAN.md** → Section "Implementation Timeline" (lines 296-549)
- **QUICK_REFERENCE.md** → Section "Implementation Priority Order"

## Assignment Components

### 1. Implementation (25 Marks)
- ✓ DeepFool algorithm - `src/deepfool.py`
- ✓ UAP algorithm - `src/uap.py`
- ✓ Constraint enforcement (L∞ or L2 norm)
- Use 500-1000 training images
- Achieve ≥70% fooling rate on test set

### 2. Evaluation (10 Marks)
- Report fooling rate on CIFAR-100 test set
- Visualize perturbation `v`
- Analyze: Does it show structural patterns or random noise?
- Compare with FGSM and PGD attacks

### 3. Analysis & Intuition ⭐ CRITICAL
**Question 1:** Why does a single vector fool majority of images?
- Explain shared subspaces of decision boundaries
- Analyze role of input space dimensionality
- **See IMPLEMENTATION_PLAN.md lines 318-422 for experimental templates**

**Question 2:** Compare UAP vs FGSM vs PGD
- Vulnerability exploited (geometry)
- Transferability hypothesis and validation
- **See IMPLEMENTATION_PLAN.md lines 424-634 for experimental design**

## Expected Results

| Metric | Expected Value |
|--------|----------------|
| UAP Fooling Rate | 70-85% |
| Perturbation Norm (L∞) | ≤ 10/255 |
| Gradient Correlation | 0.3-0.6 |
| Effective Dimension | <10% of input |
| Transfer Rate | 50-70% |

## Implementation Progress

### ✓ Completed (Phase 1: Core Setup)
- [x] Project structure and configuration
- [x] Model utilities (`src/models.py`)
- [x] Data utilities (`src/data_utils.py`)
- [x] DeepFool algorithm (`src/deepfool.py`)
- [x] UAP algorithm (`src/uap.py`)
- [x] Main execution script (`main.py`)
- [x] Pre-trained models (ResNet-32, VGG-16)

### ✓ Completed (Phase 2: Baseline Attacks)
- [x] FGSM attack implementation (`src/fgsm.py`)
- [x] PGD attack implementation (`src/pgd.py`)
- [x] Attack comparison utility (`compare_attacks.py`)
- [x] All attacks tested and validated

### ✓ Completed (Phase 3: Analysis Framework)
- [x] Evaluation utilities (`src/evaluation.py`)
- [x] Analysis experiments (`src/analysis.py`)
  - [x] Gradient correlation analysis
  - [x] PCA dimensionality analysis
  - [x] Transferability testing
- [x] Visualization generation
- [x] Comprehensive analysis script (`analyze_uap.py`)
- [x] Full pipeline script (`run_full_pipeline.py`)

### 🚧 TODO (Final Steps)
- [ ] Run full UAP computation (1000 images)
- [ ] Execute comprehensive analysis
- [ ] Generate all visualizations
- [ ] Write final report

**Next Steps:** Run full UAP computation on 1000 images, then execute comprehensive analysis.

## Documentation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **IMPLEMENTATION_PLAN.md** | Comprehensive guide with algorithms, pseudocode, analysis templates | When implementing or debugging |
| **QUICK_REFERENCE.md** | Checklist, commands, expected results | For quick lookup during work |
| **config.py** | All hyperparameters and settings | To adjust parameters |
| **This README** | Project navigation and overview | Starting point |

## References

1. Moosavi-Dezfooli, S. M., Fawzi, A., Fawzi, O., & Frossard, P. (2017). "Universal adversarial perturbations." CVPR 2017.
2. Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). "DeepFool: a simple and accurate method to fool deep neural networks." CVPR 2016.

## Time Estimates

| Phase | Task | CPU | GPU |
|-------|------|-----|-----|
| Setup | Install + Download | 10 min | 10 min |
| Model | Train ResNet-18 (100 epochs) | 8-10 hrs | 1.5-2 hrs |
| Core | Compute UAP (1000 images) | 4-6 hrs | 30-60 min |
| Analysis | All experiments | 3-4 hrs | 1-2 hrs |
| **Total** | **(with training)** | **15-24 hrs** | **5-8 hrs** |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model accuracy < 70% | Train longer or use pre-trained model |
| UAP fooling rate < 50% | Increase `xi` or `max_iter_uni` in config.py |
| Out of memory | Reduce `subset_size` in config.py |
| DeepFool not converging | Increase `max_iter` in DEEPFOOL_CONFIG |

For detailed troubleshooting, see IMPLEMENTATION_PLAN.md "Key Challenges & Solutions"

## References

1. Moosavi-Dezfooli, S. M., Fawzi, A., Fawzi, O., & Frossard, P. (2017). "Universal adversarial perturbations." CVPR 2017.
2. Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). "DeepFool: a simple and accurate method to fool deep neural networks." CVPR 2016.

## License

This project is for academic purposes only.

## Status

**Phase 1: Core Implementation** ✓ Complete
**Phase 2: Baseline Attacks** ✓ Complete
**Phase 3: Analysis Framework** ✓ Complete

**Ready to run:** Full UAP computation (1000 images) + comprehensive analysis

Current configuration:
- Device: MPS (Apple Silicon GPU)
- Source Model: ResNet-32 (70.14% accuracy)
- Target Model: VGG-16 (74.00% accuracy)
- Subset size: 1000 images
- Perturbation budget: ξ = 10/255 (L∞)

**Next step:** Execute `./run_full_pipeline.py` or `python main.py --compute-uap`