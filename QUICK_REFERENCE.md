# UAP Assignment - Quick Reference Guide

## Assignment Breakdown

### Total Points: 35 Marks
- **Implementation (25 marks):** Working UAP algorithm
- **Evaluation (10 marks):** Results, visualizations, comparisons

### Hidden Points: Analysis & Intuition
- **Critical for high marks:** These questions likely worth significant points
- Must demonstrate deep understanding, not just implementation

---

## Core Requirements Checklist

### ✓ Must Implement

1. **DeepFool Algorithm**
   - Finds minimal perturbation for single image
   - Iteratively approaches decision boundary
   - Returns perturbation vector

2. **UAP Algorithm**
   - Uses subset of 500-1000 training images
   - Iteratively aggregates DeepFool perturbations
   - Projects to norm constraint (L∞ or L2)
   - Returns universal perturbation v

3. **Model**
   - Pre-trained ResNet-18 OR VGG-16
   - Trained on CIFAR-100
   - Achieves >70% accuracy on clean images

4. **Evaluation**
   - Apply v to entire CIFAR-100 test set
   - Report fooling rate (target: ≥70%)
   - Visualize perturbation v
   - Analyze: structure vs random noise?

---

## Critical Analysis Questions (DON'T SKIP!)

### Question 1: Geometric Intuition
**"Why does a single vector fool the majority of images?"**

**What to explain:**
- Shared subspaces of decision boundaries
- How dimensionality contributes

**What to implement:**
- Gradient correlation analysis
- PCA on decision boundaries
- Dimensionality measurement

**Expected finding:**
- Gradients are correlated (>0.4)
- Low effective dimensionality (<10% of input)
- This enables universal perturbations

### Question 2: Comparative Analysis
**"Compare UAP vs FGSM vs PGD"**

**What to analyze:**
- Vulnerability exploited (geometry)
- Transferability hypothesis

**What to implement:**
- FGSM attack
- PGD attack
- Transferability test (ResNet → VGG or vice versa)

**Expected finding:**
- UAP transfers better than per-image attacks
- Because UAP exploits shared geometry, not model-specific

---

## Implementation Priority Order

### Phase 1: Core (Days 1-2)
1. Setup environment
2. Get/train CIFAR-100 model
3. Implement DeepFool
4. Implement UAP
5. **Milestone:** Achieve 70%+ fooling rate

### Phase 2: Comparison (Day 3)
1. Implement FGSM
2. Implement PGD
3. Run all three on test set
4. Generate comparison table

### Phase 3: Analysis (Day 4) ⭐ MOST IMPORTANT
1. Gradient correlation experiments
2. PCA dimensionality analysis
3. Train/get second model architecture
4. Transferability experiments
5. Generate all analysis figures

### Phase 4: Report (Day 5)
1. Write methodology
2. Present results
3. **Analysis & Intuition section** (cite experimental evidence!)
4. Discussion and conclusion

---

## Key Hyperparameters

```python
# UAP Configuration
SUBSET_SIZE = 1000          # Training images for UAP
XI = 10/255                 # Perturbation budget (L∞)
DELTA = 0.8                 # Target fooling rate (80%)
MAX_ITER_UNI = 10          # UAP iterations
NORM_TYPE = 'inf'           # 'inf' or 2

# DeepFool Configuration
MAX_ITER_DF = 50           # DeepFool iterations
OVERSHOOT = 0.02           # Boundary crossing safety

# Baseline Attacks
EPSILON = 10/255           # FGSM epsilon
PGD_ALPHA = 2/255          # PGD step size
PGD_ITERS = 10             # PGD iterations
```

---

## Expected Results

| Metric | Expected Value |
|--------|----------------|
| UAP Fooling Rate | 70-85% |
| FGSM Success Rate | 90-95% |
| PGD Success Rate | 95-99% |
| Gradient Correlation | 0.3-0.6 |
| Effective Dimension | <300 (out of 3072) |
| UAP Transfer Rate | 50-70% |
| FGSM Transfer Rate | 30-50% |
| PGD Transfer Rate | 25-40% |

---

## Required Figures (Minimum)

### Core Evaluation
1. ✓ Universal perturbation visualization (3-channel image)
2. ✓ Example images: original vs perturbed (5-10 samples)
3. ✓ Convergence plot (fooling rate vs iteration)
4. ✓ Comparison table (UAP vs FGSM vs PGD)

### Analysis (Question 1)
5. ✓ Gradient correlation heatmap
6. ✓ PCA explained variance plot
7. ✓ Effective dimensionality visualization

### Analysis (Question 2)
8. ✓ Transferability comparison bar chart
9. ✓ Transfer rate table

### Optional but Recommended
10. Perturbation frequency analysis (FFT)
11. Per-channel perturbation visualization
12. Confusion matrix (original vs perturbed)
13. Per-class vulnerability analysis

---

## Common Pitfalls to Avoid

### ❌ Implementation Mistakes
1. **Not normalizing inputs:** Ensure CIFAR-100 normalization matches training
2. **Wrong gradient computation:** Use `retain_graph=True` in DeepFool
3. **Forgetting projection:** Always project v after each update
4. **Not setting eval mode:** Model must be in `model.eval()`
5. **Memory issues:** Process images one at a time in UAP loop

### ❌ Analysis Mistakes
1. **Skipping analysis questions:** These are critical for high marks!
2. **No experimental evidence:** Don't just explain theoretically, show data
3. **Not testing transferability:** Must test on different architecture
4. **Weak interpretation:** Connect findings to theory

### ❌ Report Mistakes
1. **Missing analysis section:** Dedicates section to answering required questions
2. **No figure references:** Every claim should cite a figure/table
3. **Copy-pasting code:** Report should explain, not list code
4. **Ignoring patterns in perturbation:** Must discuss structure vs noise

---

## Quick Validation Checklist

Before submission, verify:

### Code Works
- [ ] Model accuracy >70% on clean CIFAR-100
- [ ] DeepFool fools individual images
- [ ] UAP achieves ≥70% fooling rate on test set
- [ ] Perturbation norm ≤ ξ
- [ ] FGSM and PGD implemented and working

### Analysis Complete
- [ ] Gradient correlation computed and visualized
- [ ] PCA analysis done, effective dimension reported
- [ ] Second model architecture obtained
- [ ] Transferability tested for all three attacks
- [ ] Transfer rates computed and compared

### Report Complete
- [ ] All required figures included
- [ ] Question 1 answered with experimental evidence
- [ ] Question 2 answered with transferability results
- [ ] Hypothesis stated and validated
- [ ] Interpretation of UAP structure (noise vs pattern)

### Submission Package
- [ ] All source code (clean, documented)
- [ ] requirements.txt
- [ ] README with reproduction instructions
- [ ] results/ folder with all figures
- [ ] Saved model checkpoint
- [ ] Saved UAP perturbation (v.pt)
- [ ] Report PDF

---

## Time Estimates

| Task | Estimated Time |
|------|----------------|
| Environment setup | 30 min |
| Model preparation | 1-2 hours |
| DeepFool implementation | 2-3 hours |
| UAP implementation | 3-4 hours |
| UAP computation | 1-2 hours |
| FGSM/PGD implementation | 2-3 hours |
| **Analysis experiments** | **3-4 hours** ⭐ |
| Visualization generation | 1-2 hours |
| Report writing | 3-4 hours |
| **Total** | **18-25 hours** |

---

## Resources Needed

### Computational
- **GPU:** Highly recommended (speeds up 10-20x)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** ~2GB for dataset and checkpoints
- **Runtime:** 1-2 hours for full UAP computation (GPU)

### Software
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.60.0
pandas>=1.2.0
```

### Pre-trained Models
- **Option 1:** Train ResNet-18 yourself (~2 hours on GPU)
- **Option 2:** Find pre-trained CIFAR-100 checkpoint online
- **Option 3:** Fine-tune ImageNet model

---

## Grading Expectations (Estimated)

### Implementation (25 marks)
- Correct DeepFool: 8 marks
- Correct UAP: 12 marks
- Proper norm constraint: 3 marks
- Code quality: 2 marks

### Evaluation (10 marks)
- Fooling rate ≥70%: 5 marks
- Visualizations: 3 marks
- Interpretation: 2 marks

### Analysis & Intuition (Hidden points, likely 10-15 marks)
- Question 1 with evidence: 5-7 marks
- Question 2 with experiments: 5-7 marks
- Discussion quality: 1-2 marks

**Key insight:** Analysis questions likely worth significant points!
Don't just implement - demonstrate understanding!

---

## Quick Start Commands

```bash
# Setup
pip install -r requirements.txt

# Train model (if needed)
python train_model.py --model resnet18 --dataset cifar100 --epochs 100

# Compute UAP
python main.py --compute-uap --subset-size 1000 --xi 0.039 --max-iter 10

# Evaluate
python main.py --evaluate --uap-path results/uap.pt

# Run analysis experiments
python main.py --analyze --source-model resnet18 --target-model vgg16

# Generate all figures
python main.py --visualize
```

---

## Final Tips

1. **Start early:** 18-25 hours of work, don't underestimate
2. **Test incrementally:** Verify DeepFool works before UAP
3. **Save everything:** Checkpoints, intermediate results, logs
4. **Document as you go:** Write methodology while implementing
5. **Run analysis early:** May reveal interesting findings for discussion
6. **Multiple iterations:** First UAP might not reach 70%, tune hyperparameters
7. **Backup code:** Use git or regular backups
8. **Ask for help:** If stuck, consult paper, peers, or instructor

**Most Important:** The analysis questions are not afterthoughts!
They test your understanding and are likely worth significant marks.
Budget adequate time for experiments and thoughtful interpretation.

Good luck! 🚀
