# Google Colab Execution Guide

## Quick Start

### Option 1: Direct Upload (Recommended)
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File → Upload notebook`
3. Upload `UAP_Full_Experiment.ipynb` from this repository
4. Click `Runtime → Change runtime type → GPU (T4 or better)`
5. Run all cells: `Runtime → Run all`

### Option 2: From GitHub (If Public)
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File → Open notebook → GitHub`
3. Enter repository URL
4. Select `UAP_Full_Experiment.ipynb`
5. Enable GPU and run all cells

### Option 3: From Private Repository
Since the repository is on Salesforce's internal Git:
1. Clone locally: `git clone git@git.soma.salesforce.com:mkanaka/rsai-a1.git`
2. Upload `UAP_Full_Experiment.ipynb` to Colab (Option 1)

## What the Notebook Does

The notebook runs the complete experiment pipeline:

### 1. Setup (5 minutes)
- ✓ Checks GPU availability
- ✓ Clones repository (if needed)
- ✓ Installs dependencies
- ✓ Loads pre-trained models (ResNet-32, VGG-16)
- ✓ Verifies model accuracy (≥70%)

### 2. UAP Computation (1-2 hours on GPU)
- ✓ Computes Universal Adversarial Perturbation
- ✓ Uses 1000 training images from CIFAR-100
- ✓ Target: ≥70% fooling rate
- ✓ Saves checkpoints every iteration

### 3. Question 1 Analysis (10-15 minutes)
- ✓ Gradient correlation analysis (100 samples)
- ✓ PCA dimensionality analysis (500 samples)
- ✓ Generates correlation matrix visualization
- ✓ Generates PCA variance plots

### 4. Question 2 Analysis (5-10 minutes)
- ✓ Transferability testing (ResNet-32 → VGG-16)
- ✓ Tests on 1000 test images
- ✓ Generates adversarial examples visualization

### 5. Results Export
- ✓ Saves all visualizations as PNG files
- ✓ Exports metrics as JSON
- ✓ Creates downloadable ZIP archive

## Expected Results

| Metric | Expected Value |
|--------|----------------|
| UAP Fooling Rate | 70-85% |
| Perturbation L∞ norm | ≤ 10/255 ≈ 0.0392 |
| Gradient Correlation | 0.2-0.6 |
| Effective Dimension | <20 components (of 50) |
| Transferability Ratio | 0.5-0.9 |

## Runtime Breakdown

### With GPU (T4)
- Setup: 5 min
- UAP Computation: 1-2 hours
- Analysis: 15-20 min
- **Total: ~2 hours**

### Without GPU (CPU only)
- Setup: 5 min
- UAP Computation: 6-8 hours ⚠️
- Analysis: 30-40 min
- **Total: ~8 hours** (Not recommended)

## GPU Recommendations

### Free Tier GPUs (Colab)
- ✓ **T4** (16GB VRAM) - Good for this project
- ✓ **K80** (12GB VRAM) - Acceptable but slower

### Colab Pro GPUs
- ✓ **V100** (16GB VRAM) - Faster (~1 hour total)
- ✓ **A100** (40GB VRAM) - Fastest (~45 min total)

## Monitoring Progress

The notebook includes progress bars for:
- UAP iterations (10 total)
- Per-image DeepFool computation
- Gradient computation for analysis
- Transferability testing

Each cell displays:
- Current iteration/sample
- Fooling rate
- Time estimates

## Downloading Results

After completion, the notebook creates `uap_experiment_results.zip` containing:

```
results/
├── uap_final.pt                   # Trained UAP
├── uap_perturbation.png           # UAP visualization
├── uap_convergence.png            # Training curve
├── gradient_correlation.png       # Question 1
├── pca_variance.png               # Question 1
├── adversarial_examples.png       # Example attacks
└── experiment_results.json        # All metrics
```

**To download:**
1. Click the folder icon (📁) in left sidebar
2. Navigate to `results/`
3. Right-click `uap_experiment_results.zip`
4. Select "Download"

## Troubleshooting

### Issue: "GPU not available"
**Solution:**
- Go to `Runtime → Change runtime type`
- Set Hardware accelerator to "GPU"
- Restart runtime

### Issue: "Session disconnected"
**Solution:**
- Colab free tier has ~12 hour limit
- If disconnected, restart and run from checkpoint
- UAP checkpoints saved every iteration in `./results/`

### Issue: "Out of memory"
**Solution:**
- Reduce batch sizes in the notebook
- Reduce analysis sample sizes
- Use Colab Pro for more memory

### Issue: "Repository clone fails"
**Solution:**
- Since repo is on internal Salesforce Git, use Option 1 (Upload notebook)
- Or manually upload the `src/` folder and required files

### Issue: "Model download slow"
**Solution:**
- First time downloads pre-trained models from torch.hub
- Models are cached for future runs
- Takes 5-10 minutes on first run

## Alternative: Local Execution

If you have a local GPU, you can run locally instead:

```bash
# On your machine with GPU
cd rsai-a1
source venv/bin/activate

# Run equivalent to notebook
python main.py --compute-uap
python analyze_uap.py --uap-path ./results/uap_final.pt
```

## Files Included in Notebook

The notebook automatically handles:
- ✓ All source files (`src/*.py`)
- ✓ Configuration (`config.py`)
- ✓ Model downloads (from torch.hub)
- ✓ Dataset downloads (CIFAR-100 auto-download)

No manual file setup needed!

## Tips for Best Results

1. **Use GPU**: CPU mode takes 4x longer
2. **Don't close browser**: Keep Colab tab open during computation
3. **Monitor memory**: Check GPU usage in Colab sidebar
4. **Save frequently**: Download checkpoints periodically
5. **Pro subscription**: Consider for longer runtime limits

## Next Steps

After downloading results:
1. Review all visualizations
2. Check `experiment_results.json` for exact metrics
3. Use findings to write your report
4. Include visualizations in report figures

## Questions?

Refer to main documentation:
- `README.md` - Project overview
- `IMPLEMENTATION_PLAN.md` - Detailed algorithms
- `QUICK_REFERENCE.md` - Command reference

---

**Ready to run?** Upload `UAP_Full_Experiment.ipynb` to Colab and click "Run all"! 🚀
