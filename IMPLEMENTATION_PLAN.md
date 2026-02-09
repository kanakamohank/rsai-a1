# Universal Adversarial Perturbations (UAP) Implementation Plan

## Overview
Implementation of Universal Adversarial Perturbations following Moosavi-Dezfooli et al. (CVPR 2017) for CIFAR-100 dataset.

## Algorithm Overview

### Core Concept
A UAP is a single perturbation vector `v` that can fool a neural network on most images from a distribution, rather than being image-specific like FGSM.

### Key Equation
`k̂(x + v) ≠ k̂(x)` for most `x ~ D`

Subject to: `||v||_p ≤ ξ`

## Implementation Plan

### Phase 1: Environment Setup (Prerequisites)

#### 1.1 Dependencies
```
- PyTorch (≥1.9.0)
- torchvision
- numpy
- matplotlib
- tqdm (for progress bars)
```

#### 1.2 Project Structure
```
rsai-a1/
├── src/
│   ├── __init__.py
│   ├── models.py          # Model loading and preparation
│   ├── uap.py             # Core UAP algorithm implementation
│   ├── deepfool.py        # DeepFool algorithm (used by UAP)
│   └── evaluation.py      # Evaluation metrics and visualization
├── data/                  # CIFAR-100 dataset (auto-downloaded)
├── checkpoints/           # Pre-trained models
├── results/               # Generated perturbations and visualizations
├── main.py               # Main execution script
├── requirements.txt
└── README.md
```

### Phase 2: Core Algorithm Implementation

#### 2.1 DeepFool Algorithm
The UAP algorithm iteratively uses DeepFool to find minimal perturbations. DeepFool finds the closest decision boundary.

**DeepFool Steps:**
1. For input image x, compute the model's output
2. Find the closest decision boundary by iteratively:
   - Computing gradients w.r.t. each class
   - Finding the minimal perturbation to reach the closest class boundary
   - Updating x until misclassification occurs
3. Return the minimal perturbation

**Key Parameters:**
- `max_iter`: Maximum iterations (default: 50)
- `overshoot`: Small constant to ensure boundary crossing (default: 0.02)

#### 2.2 UAP Algorithm (Main Implementation)

**Algorithm Pseudocode:**
```
Input:
  - X: set of training images
  - f: classifier (neural network)
  - ξ: perturbation magnitude constraint
  - δ: desired fooling rate
  - max_iter_uni: maximum iterations

Initialize: v = 0

while fooling_rate(v) < δ and iter < max_iter_uni:
    for each x_i in X:
        if f(x_i + v) == f(x_i):  # if not yet fooled
            # Find minimal perturbation using DeepFool
            Δv_i = DeepFool(x_i + v, f)

            # Update universal perturbation
            v = v + Δv_i

            # Project v to satisfy constraint ||v||_p ≤ ξ
            v = project(v, ξ, p)

    # Check fooling rate on all images
    fooling_rate = compute_fooling_rate(X, v, f)

return v
```

**Key Implementation Details:**

1. **Projection Step:**
   - L∞ norm: `v = clip(v, -ξ, ξ)`
   - L2 norm: `v = v * min(1, ξ / ||v||_2)`

2. **Efficient Batch Processing:**
   - Process images in batches where possible
   - Use GPU acceleration

3. **Checkpointing:**
   - Save intermediate v at each iteration
   - Enable resuming from checkpoints

### Phase 3: Model Setup

#### 3.1 Model Selection
Use either:
- **ResNet-18** (recommended): Good balance of accuracy and speed
- **VGG-16**: Deeper architecture, potentially more vulnerable

#### 3.2 Pre-trained Model
Options:
1. Train from scratch on CIFAR-100 (takes time)
2. Use pre-trained model (faster, recommended for assignment)
3. Fine-tune ImageNet pre-trained model for CIFAR-100

**Model Preparation:**
```python
import torch
import torchvision.models as models

# Load model
model = models.resnet18(pretrained=False, num_classes=100)

# Load CIFAR-100 trained weights (need to train or download)
# model.load_state_dict(torch.load('checkpoints/resnet18_cifar100.pth'))

model.eval()
model.to(device)
```

### Phase 4: Dataset Preparation

#### 4.1 CIFAR-100 Setup
```python
from torchvision import datasets, transforms

# Normalization values for CIFAR-100
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Training subset for UAP computation (500-1000 images)
train_dataset = datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Full test set for evaluation
test_dataset = datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
```

#### 4.2 Subset Selection
```python
import random

# Select random subset for UAP computation
subset_size = 1000
indices = random.sample(range(len(train_dataset)), subset_size)
subset = torch.utils.data.Subset(train_dataset, indices)
```

### Phase 5: Implementation Details

#### 5.1 Key Hyperparameters

| Parameter | Description | Suggested Value |
|-----------|-------------|-----------------|
| `xi` | Perturbation magnitude | 10/255 for L∞, 10 for L2 |
| `delta` | Desired fooling rate | 0.8 (80%) |
| `max_iter_uni` | Max UAP iterations | 10-20 |
| `max_iter_df` | Max DeepFool iterations | 50 |
| `p` | Norm type | 'inf' or 2 |
| `num_classes` | CIFAR-100 classes | 100 |

#### 5.2 Computational Considerations

1. **Memory Management:**
   - Process images one at a time in DeepFool
   - Batch evaluation for fooling rate
   - Use mixed precision (fp16) if needed

2. **Speed Optimization:**
   - Use GPU (CUDA)
   - Limit DeepFool iterations
   - Early stopping if fooling rate reaches target

3. **Convergence:**
   - Monitor fooling rate per iteration
   - Save best perturbation
   - Plot convergence curve

### Phase 6: Evaluation (10 Marks)

#### 6.1 Fooling Rate Computation
```python
def compute_fooling_rate(model, dataset, perturbation):
    """
    Compute percentage of images where prediction changes
    """
    fooled = 0
    total = 0

    for image, label in dataset:
        # Original prediction
        orig_pred = model(image).argmax()

        # Perturbed prediction
        pert_pred = model(image + perturbation).argmax()

        if orig_pred != pert_pred:
            fooled += 1
        total += 1

    return fooled / total * 100
```

#### 6.2 Metrics to Report

1. **Primary Metric:**
   - Fooling Rate on CIFAR-100 Test Set (%)

2. **Secondary Metrics:**
   - Original model accuracy
   - Perturbed model accuracy
   - Perturbation norm (L2 and L∞)
   - Top-5 accuracy comparison

3. **Per-Class Analysis:**
   - Which classes are most vulnerable?
   - Confusion matrix visualization

#### 6.3 Comparative Analysis with FGSM and PGD

**IMPORTANT:** You must implement and compare UAP against per-image attacks:

1. **FGSM (Fast Gradient Sign Method):**
```python
def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM: perturb = epsilon * sign(gradient)
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
```

2. **PGD (Projected Gradient Descent):**
```python
def pgd_attack(model, image, label, epsilon, alpha, num_iter):
    """
    PGD: Iterative FGSM with projection
    """
    perturbed_image = image.clone()

    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.cross_entropy(output, label)

        model.zero_grad()
        loss.backward()

        # Update
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
            # Project back to epsilon ball
            perturbation = torch.clamp(perturbed_image - image, -epsilon, epsilon)
            perturbed_image = torch.clamp(image + perturbation, 0, 1)

    return perturbed_image
```

**Comparison Metrics:**
- Attack Success Rate for each method
- Average perturbation norm
- Computation time (total for dataset)
- Perceptual quality (SSIM, PSNR)

#### 6.4 Visualization Requirements

1. **Perturbation Visualization:**
```python
# Visualize v
plt.imshow(perturbation.permute(1, 2, 0))
plt.title('Universal Perturbation')
plt.colorbar()
plt.savefig('results/uap_visualization.png')
```

2. **Spectral Analysis:**
   - FFT of perturbation
   - Check for structural patterns vs random noise

3. **Example Images:**
   - Show original images
   - Show perturbed images
   - Display predictions before/after

4. **Convergence Plot:**
   - Fooling rate vs iteration
   - Perturbation norm vs iteration

#### 6.5 Required Analysis & Intuition (Critical for Report)

You MUST address these questions in your report with experimental evidence:

**Question 1: Geometric Intuition - Why Single Vector Fools Majority of Images?**

You need to explain and demonstrate:

1. **Shared Subspaces of Decision Boundaries:**
   - What are decision boundaries in DNNs?
   - Why do different images share similar vulnerabilities?
   - Concept: Normal vectors to decision boundaries are correlated

   **Experiments to conduct:**
   ```python
   # 1. Analyze decision boundary normals for multiple images
   # Compute gradient of loss w.r.t. input for different images
   # Measure correlation between gradients

   def compute_gradient_correlation(model, images):
       """
       Measure how correlated the gradients are across images
       High correlation → shared vulnerability directions
       """
       gradients = []
       for img in images:
           img.requires_grad = True
           output = model(img)
           loss = output.max()
           loss.backward()
           gradients.append(img.grad.flatten())

       # Compute pairwise correlation
       correlation_matrix = np.corrcoef(gradients)
       return correlation_matrix
   ```

   **Key points to address:**
   - If gradients are highly correlated (>0.5), shared vulnerability exists
   - UAP exploits this correlation by finding a common direction
   - Visualize: Heatmap of gradient correlations

2. **Role of Input Space Dimensionality:**
   - CIFAR-100: 32×32×3 = 3,072 dimensions
   - High-dimensional spaces have counter-intuitive geometry

   **The "Curse of Dimensionality" paradox:**
   - In high dimensions, most directions are nearly orthogonal
   - BUT decision boundaries have low effective dimensionality
   - Networks learn lower-dimensional manifolds

   **Experiments to conduct:**
   ```python
   # 2. Measure effective dimensionality of decision boundaries
   # Use PCA on gradients

   def analyze_gradient_dimensionality(model, images):
       """
       How many principal components capture most variance?
       Low effective dimension → easier to find universal perturbation
       """
       from sklearn.decomposition import PCA

       gradients = [compute_gradient(model, img) for img in images]
       gradients = np.array([g.flatten().numpy() for g in gradients])

       pca = PCA()
       pca.fit(gradients)

       # Plot explained variance ratio
       plt.plot(np.cumsum(pca.explained_variance_ratio_))
       plt.xlabel('Number of Components')
       plt.ylabel('Cumulative Explained Variance')
       plt.title('Effective Dimensionality of Decision Boundaries')

       # Report: How many components explain 90% variance?
       n_components_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9)
       return n_components_90
   ```

   **Key points to address:**
   - If 90% variance captured by <100 components out of 3072
   - Decision boundary lives in low-dimensional subspace
   - UAP can exploit this low-dimensional structure
   - Formula: If decision boundary has effective dimension d << D (input dimension)
     - Probability of finding universal direction increases

3. **Why Universal Perturbations Exist - Mathematical Insight:**

   **To demonstrate in report:**
   - Show that decision boundary normals cluster in certain directions
   - Prove experimentally that small perturbation in clustered direction affects many images
   - Visualize: t-SNE or PCA of gradients showing clustering

   **Expected observations:**
   - High correlation between gradients (>0.4)
   - Low effective dimensionality (<10% of input dimension)
   - UAP aligns with principal components of gradient distribution

**Question 2: Comparative Analysis - UAP vs FGSM vs PGD**

You need to compare three attack methods systematically:

**Part A: Vulnerability Exploited - Geometric Differences**

| Attack | Geometry Exploited | Optimization | Adaptivity |
|--------|-------------------|--------------|------------|
| **UAP** | Shared vulnerabilities across images | Aggregates per-image perturbations | Image-agnostic |
| **FGSM** | Local gradient direction for single image | One-step, sign of gradient | Image-specific |
| **PGD** | Optimal perturbation for single image | Iterative, projected gradient | Image-specific |

**Experiments to conduct:**

1. **Attack Success Rate Comparison:**
```python
results = {
    'UAP': {
        'fooling_rate': 0.0,
        'avg_perturbation_norm': 0.0,
        'total_time': 0.0,
        'perturbations_computed': 1,  # Single universal
    },
    'FGSM': {
        'fooling_rate': 0.0,
        'avg_perturbation_norm': 0.0,
        'total_time': 0.0,
        'perturbations_computed': 10000,  # Per-image
    },
    'PGD': {
        'fooling_rate': 0.0,
        'avg_perturbation_norm': 0.0,
        'total_time': 0.0,
        'perturbations_computed': 10000,  # Per-image
    }
}
```

2. **Perturbation Characteristics:**
```python
# Visualize perturbations side-by-side
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Original images
# Row 2: Perturbed images
# Col 1: UAP | Col 2: FGSM | Col 3: PGD

# Show that UAP is universal while FGSM/PGD are image-specific
```

3. **Geometric Analysis:**
```python
# For a sample of images, compute:
# - Direction of UAP perturbation
# - Direction of FGSM perturbation
# - Direction of PGD perturbation
# - Measure angle between perturbations

def compute_perturbation_angles(uap, fgsm_perts, pgd_perts):
    """
    Measure angle between UAP and per-image perturbations
    Large angles → different geometric vulnerabilities exploited
    """
    angles_fgsm = []
    angles_pgd = []

    for fgsm_p, pgd_p in zip(fgsm_perts, pgd_perts):
        # Cosine similarity
        angle_fgsm = cosine_similarity(uap.flatten(), fgsm_p.flatten())
        angle_pgd = cosine_similarity(uap.flatten(), pgd_p.flatten())

        angles_fgsm.append(angle_fgsm)
        angles_pgd.append(angle_pgd)

    return angles_fgsm, angles_pgd
```

**Key points to address in report:**

a) **FGSM:**
   - Exploits: Local linear approximation of loss
   - Geometry: Moves in direction of steepest ascent (gradient)
   - Limitation: Single step, may not reach decision boundary
   - Characteristic: Image-specific, fast but sub-optimal

b) **PGD:**
   - Exploits: Finds optimal perturbation within epsilon ball
   - Geometry: Iteratively climbs loss landscape with projection
   - Advantage: Stronger than FGSM, near-optimal for per-image
   - Characteristic: Image-specific, slower but more effective

c) **UAP:**
   - Exploits: Correlated normal vectors across decision boundaries
   - Geometry: Finds direction that crosses boundaries for MANY images
   - Advantage: Single perturbation for entire dataset
   - Limitation: Lower success rate per image than PGD
   - Characteristic: Image-agnostic, powerful for practical attacks

**Part B: Transferability Hypothesis**

You need to test and explain which attack transfers better to different architecture.

**Hypothesis to test:**
> "UAP should be MORE transferable than per-image attacks because it exploits
> fundamental geometric properties shared across architectures, rather than
> overfitting to specific decision boundaries of one model."

**Experimental Design:**

1. **Source Model:** ResNet-18 trained on CIFAR-100
2. **Target Model:** VGG-16 trained on CIFAR-100
3. **Generate attacks on ResNet-18:**
   - UAP (single perturbation)
   - FGSM (per-image perturbations)
   - PGD (per-image perturbations)
4. **Test on VGG-16:**
   - Measure fooling rate for each attack
   - Report transferability rate

```python
def test_transferability(source_model, target_model, test_data):
    """
    Generate attacks on source_model
    Test on target_model
    Report transfer rates
    """

    # Generate attacks on source model
    uap = compute_uap(source_model, train_subset)

    fgsm_success_source = 0
    fgsm_success_target = 0
    pgd_success_source = 0
    pgd_success_target = 0
    uap_success_source = 0
    uap_success_target = 0

    for image, label in test_data:
        # Generate per-image attacks
        fgsm_pert = fgsm_attack(source_model, image, epsilon)
        pgd_pert = pgd_attack(source_model, image, label, epsilon)

        # Test on source
        if source_model(image + fgsm_pert).argmax() != label:
            fgsm_success_source += 1
        if source_model(image + pgd_pert).argmax() != label:
            pgd_success_source += 1
        if source_model(image + uap).argmax() != label:
            uap_success_source += 1

        # Test on target
        if target_model(image + fgsm_pert).argmax() != label:
            fgsm_success_target += 1
        if target_model(image + pgd_pert).argmax() != label:
            pgd_success_target += 1
        if target_model(image + uap).argmax() != label:
            uap_success_target += 1

    # Compute transfer rates
    transfer_rates = {
        'FGSM': fgsm_success_target / fgsm_success_source,
        'PGD': pgd_success_target / pgd_success_source,
        'UAP': uap_success_target / uap_success_source,
    }

    return transfer_rates
```

**Expected Results Table:**

| Attack | Success on Source (%) | Success on Target (%) | Transfer Rate (%) |
|--------|----------------------|----------------------|-------------------|
| FGSM   | ~95-100              | ~30-50               | ~30-50            |
| PGD    | ~95-100              | ~25-40               | ~25-40            |
| UAP    | ~75-85               | ~40-60               | ~50-70            |

**Justification to provide in report:**

1. **Why UAP transfers better:**
   - Exploits geometric properties shared across architectures
   - Both ResNet and VGG learn similar low-level features
   - Decision boundaries have correlated normal vectors
   - UAP doesn't overfit to specific model's decision boundary

2. **Why per-image attacks don't transfer well:**
   - FGSM/PGD optimize for specific model's loss landscape
   - Decision boundaries differ between architectures
   - Perturbation direction is model-specific
   - Higher success rate on source → more overfitting → worse transfer

3. **Mathematical intuition:**
   - UAP finds: v such that P_{x~D}[f₁(x+v) ≠ f₁(x)] is high
   - If f₁ and f₂ have correlated decision boundaries
   - Then P_{x~D}[f₂(x+v) ≠ f₂(x)] also high
   - Per-image attacks: argmax_δ L(f₁(x+δ), y)
   - This is specific to f₁'s loss landscape

**Validation Checklist for Analysis Section:**

- [ ] Computed gradient correlation matrix (>0.4 expected)
- [ ] Performed PCA on gradients (effective dimension < 10% of input dimension)
- [ ] Implemented all three attacks (UAP, FGSM, PGD)
- [ ] Generated comparison table with metrics
- [ ] Visualized perturbations side-by-side
- [ ] Measured perturbation angles/similarity
- [ ] Tested transferability on second model architecture
- [ ] Computed transfer rates
- [ ] Provided geometric explanation for observations
- [ ] Justified transferability hypothesis with experimental evidence

### Phase 7: Expected Results & Analysis

#### 7.1 Expected Outcomes
- Fooling rate: 70-85% (depending on hyperparameters)
- Perturbation should be imperceptible to humans
- May show structural patterns (not pure noise)

#### 7.2 Analysis Points

1. **Structural Patterns:**
   - UAPs often show:
     - High-frequency components
     - Color-specific biases
     - Geometric patterns (edges, textures)
   - This suggests networks have shared vulnerabilities

2. **Transferability:**
   - Test if UAP transfers across architectures
   - ResNet-18 → VGG-16?

3. **Norm Comparison:**
   - L∞ vs L2: which is more effective?
   - Trade-off between visibility and fooling rate

## Implementation Timeline (Revised with Analysis)

### Step 1: Setup (30 mins)
- Install dependencies
- Create project structure
- Download CIFAR-100
- Set random seeds for reproducibility

### Step 2: Model Preparation (1-2 hours)
- Train or download pre-trained model
- Verify model accuracy on clean images (should be >70%)
- Test forward and backward pass
- Ensure GPU is being used

### Step 3: DeepFool Implementation (2-3 hours)
- Implement core algorithm
- Test on individual images
- Debug and validate
- Verify it finds minimal perturbation

### Step 4: UAP Implementation (3-4 hours)
- Implement main algorithm
- Add projection and checkpointing
- Test on small subset (100 images)
- Verify convergence behavior

### Step 5: Full UAP Computation (1-2 hours)
- Run on 500-1000 images
- Monitor convergence
- Save final perturbation
- Should achieve 70%+ fooling rate

### Step 6: Baseline Attacks Implementation (2-3 hours) ⭐ NEW
- Implement FGSM
- Implement PGD (with 10-20 iterations)
- Test on sample images
- Verify attack success

### Step 7: Evaluation Experiments (2-3 hours)
- Compute metrics on full test set for all attacks
- Generate comparison table
- Create visualizations
- Measure computation time

### Step 8: Analysis Experiments (3-4 hours) ⭐ CRITICAL
**For Question 1 - Geometric Intuition:**
- Compute gradients for sample of images
- Calculate gradient correlation matrix
- Perform PCA on gradients
- Generate visualizations
- Write explanation based on results

**For Question 2 - Comparative Analysis:**
- Train/obtain second model architecture (VGG-16 if started with ResNet-18)
- Test transferability of all three attacks
- Compute transfer rates
- Generate comparison visualizations
- Validate or refute hypothesis

### Step 9: Visualization Generation (1-2 hours)
- Universal perturbation visualization
- Example images (original vs perturbed)
- Convergence plots
- Frequency analysis of perturbation
- All analysis figures (correlation, PCA, transferability)

### Step 10: Report Writing (3-4 hours)
- Write methodology section
- Present results with tables and figures
- **Analysis & Intuition section** (most important!)
  - Question 1 with experimental evidence
  - Question 2 with transferability results
- Discussion and insights
- Proofread and polish

**Total Estimated Time: 18-25 hours**

## Detailed Step-by-Step Execution Plan

### Phase 1: Core Implementation (Steps 1-5)

**Day 1-2: Get UAP Working**
```
Goal: Successfully compute a universal perturbation with 70%+ fooling rate

Tasks:
1. Set up environment ✓
2. Get/train model ✓
3. Implement DeepFool ✓
4. Implement UAP ✓
5. Compute UAP on subset ✓

Success criteria:
- Model accuracy > 70% on clean images
- DeepFool successfully fools individual images
- UAP achieves 70%+ fooling rate on test set
- Perturbation norm within budget
```

### Phase 2: Comparative Implementation (Steps 6-7)

**Day 3: Implement Baseline Attacks**
```
Goal: Have FGSM and PGD working for comparison

Tasks:
1. Implement FGSM (1 hour)
2. Implement PGD (1-2 hours)
3. Test on samples
4. Run on test set
5. Generate comparison table

Success criteria:
- FGSM achieves >90% success rate
- PGD achieves >95% success rate
- Comparison metrics computed
```

### Phase 3: Analysis & Experiments (Step 8)

**Day 4: Critical Analysis Experiments**
```
Goal: Generate all evidence needed to answer analysis questions

For Question 1 (4 hours):
1. Implement gradient extraction (30 min)
2. Compute correlation matrix (30 min)
3. Perform PCA analysis (1 hour)
4. Generate visualizations (1 hour)
5. Interpret results (1 hour)

For Question 2 (3 hours):
1. Obtain second model architecture (30 min)
2. Generate attacks on source model (30 min)
3. Test on target model (1 hour)
4. Compute transfer rates (30 min)
5. Generate comparison plots (30 min)

Success criteria:
- Have gradient correlation heatmap
- Have PCA explained variance plot
- Have transferability comparison table
- Have evidence to support/refute hypothesis
```

### Phase 4: Finalization (Steps 9-10)

**Day 5: Visualization & Report**
```
Goal: Complete all deliverables

Morning (4 hours):
1. Generate all required visualizations
2. Organize results folder
3. Ensure reproducibility (README, requirements)

Afternoon (4 hours):
1. Write report draft
2. Insert all figures and tables
3. Write analysis section with evidence
4. Proofread and polish
5. Final submission package

Success criteria:
- All figures are clear and labeled
- Report addresses all questions
- Analysis section has experimental evidence
- Code is clean and documented
- README allows reproduction
```

## Pseudocode Templates for Analysis Experiments

### Template 1: Gradient Correlation Analysis (Question 1)

```python
def analyze_shared_vulnerabilities(model, dataset, num_samples=100):
    """
    Analyze why a single perturbation works for many images
    by measuring gradient correlation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Step 1: Extract gradients for sample images
    gradients = []
    images_sample = random.sample(dataset, num_samples)

    for image, label in images_sample:
        image = image.unsqueeze(0).to(device)
        image.requires_grad = True

        # Forward pass
        output = model(image)
        pred_class = output.argmax()

        # Backward pass - get gradient of predicted class score
        model.zero_grad()
        output[0, pred_class].backward()

        # Store gradient
        grad = image.grad.detach().cpu().flatten().numpy()
        gradients.append(grad)

    gradients = np.array(gradients)  # Shape: (num_samples, 3072)

    # Step 2: Compute correlation matrix
    correlation_matrix = np.corrcoef(gradients)

    # Step 3: Visualize correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True)
    plt.title('Gradient Correlation Across Images')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.savefig('results/gradient_correlation.png', dpi=300)

    # Step 4: Compute statistics
    # Exclude diagonal (self-correlation = 1)
    mask = ~np.eye(num_samples, dtype=bool)
    correlations = correlation_matrix[mask]

    mean_correlation = correlations.mean()
    median_correlation = np.median(correlations)
    high_corr_percentage = (correlations > 0.4).mean() * 100

    print(f"Mean correlation: {mean_correlation:.3f}")
    print(f"Median correlation: {median_correlation:.3f}")
    print(f"Percentage with correlation > 0.4: {high_corr_percentage:.1f}%")

    # Step 5: Interpretation for report
    interpretation = f"""
    Analysis Results:
    - Mean gradient correlation: {mean_correlation:.3f}
    - {high_corr_percentage:.1f}% of image pairs have correlation > 0.4

    Interpretation:
    {"High correlation suggests decision boundaries share common directions, " +
     "explaining why a universal perturbation can fool many images."
     if mean_correlation > 0.3 else
     "Low correlation suggests less shared vulnerability structure."}
    """

    return {
        'correlation_matrix': correlation_matrix,
        'mean_correlation': mean_correlation,
        'interpretation': interpretation
    }
```

### Template 2: Dimensionality Analysis (Question 1)

```python
def analyze_decision_boundary_dimensionality(model, dataset, num_samples=500):
    """
    Measure effective dimensionality of decision boundaries
    using PCA on gradients
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    # Step 1: Collect gradients
    gradients = []
    images_sample = random.sample(dataset, num_samples)

    for image, label in tqdm(images_sample, desc="Computing gradients"):
        image = image.unsqueeze(0).to(device)
        image.requires_grad = True

        output = model(image)
        pred_class = output.argmax()

        model.zero_grad()
        output[0, pred_class].backward()

        grad = image.grad.detach().cpu().flatten().numpy()
        gradients.append(grad)

    gradients = np.array(gradients)
    input_dimension = gradients.shape[1]  # e.g., 3072 for CIFAR

    # Step 2: Perform PCA
    pca = PCA()
    pca.fit(gradients)

    # Step 3: Compute effective dimensionality
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cumsum_variance > 0.90) + 1
    n_components_95 = np.argmax(cumsum_variance > 0.95) + 1

    # Step 4: Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cumulative explained variance
    ax1.plot(cumsum_variance[:500])
    ax1.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
    ax1.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    ax1.axvline(x=n_components_90, color='r', linestyle=':', alpha=0.5)
    ax1.axvline(x=n_components_95, color='g', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Number of Principal Components')
    ax1.set_ylabel('Cumulative Explained Variance')
    ax1.set_title('Effective Dimensionality of Decision Boundaries')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Individual explained variance (first 50 components)
    ax2.bar(range(50), pca.explained_variance_ratio_[:50])
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Individual Component Contributions')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/pca_dimensionality_analysis.png', dpi=300)

    # Step 5: Report findings
    percentage_90 = (n_components_90 / input_dimension) * 100
    percentage_95 = (n_components_95 / input_dimension) * 100

    interpretation = f"""
    Dimensionality Analysis Results:

    Input space dimension: {input_dimension}

    Effective dimension (90% variance): {n_components_90} ({percentage_90:.1f}% of input)
    Effective dimension (95% variance): {n_components_95} ({percentage_95:.1f}% of input)

    Interpretation:
    Decision boundaries live in a low-dimensional subspace ({percentage_90:.1f}% of full space).
    This explains why finding a universal perturbation is feasible - we only need to
    find a direction in this low-dimensional subspace that affects many images.

    The first {n_components_90} principal components capture most of the vulnerability
    structure. A universal perturbation can align with these principal directions.
    """

    print(interpretation)

    return {
        'n_components_90': n_components_90,
        'n_components_95': n_components_95,
        'percentage_90': percentage_90,
        'pca': pca,
        'interpretation': interpretation
    }
```

### Template 3: Transferability Analysis (Question 2)

```python
def analyze_transferability(source_model, target_model, test_dataset,
                           uap, epsilon=10/255, num_samples=1000):
    """
    Test and compare transferability of UAP vs per-image attacks
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Sample test images
    test_sample = random.sample(list(test_dataset), num_samples)

    # Initialize counters
    results = {
        'UAP': {'source_success': 0, 'target_success': 0},
        'FGSM': {'source_success': 0, 'target_success': 0},
        'PGD': {'source_success': 0, 'target_success': 0},
    }

    print("Testing transferability...")

    for image, true_label in tqdm(test_sample):
        image = image.unsqueeze(0).to(device)
        true_label = torch.tensor([true_label]).to(device)

        # Original predictions
        source_orig_pred = source_model(image).argmax()
        target_orig_pred = target_model(image).argmax()

        # Skip if models disagree on clean image (for fair comparison)
        if source_orig_pred != target_orig_pred:
            continue

        # Test UAP
        uap_perturbed = torch.clamp(image + uap, 0, 1)
        source_uap_pred = source_model(uap_perturbed).argmax()
        target_uap_pred = target_model(uap_perturbed).argmax()

        if source_uap_pred != source_orig_pred:
            results['UAP']['source_success'] += 1
            if target_uap_pred != target_orig_pred:
                results['UAP']['target_success'] += 1

        # Test FGSM
        fgsm_perturbed = fgsm_attack(source_model, image, epsilon)
        source_fgsm_pred = source_model(fgsm_perturbed).argmax()
        target_fgsm_pred = target_model(fgsm_perturbed).argmax()

        if source_fgsm_pred != source_orig_pred:
            results['FGSM']['source_success'] += 1
            if target_fgsm_pred != target_orig_pred:
                results['FGSM']['target_success'] += 1

        # Test PGD
        pgd_perturbed = pgd_attack(source_model, image, true_label,
                                   epsilon, alpha=2/255, num_iter=10)
        source_pgd_pred = source_model(pgd_perturbed).argmax()
        target_pgd_pred = target_model(pgd_perturbed).argmax()

        if source_pgd_pred != source_orig_pred:
            results['PGD']['source_success'] += 1
            if target_pgd_pred != target_orig_pred:
                results['PGD']['target_success'] += 1

    # Compute transfer rates
    transfer_data = []
    for attack_name, counts in results.items():
        source_rate = counts['source_success'] / num_samples * 100
        target_rate = counts['target_success'] / num_samples * 100
        if counts['source_success'] > 0:
            transfer_rate = counts['target_success'] / counts['source_success'] * 100
        else:
            transfer_rate = 0

        transfer_data.append({
            'Attack': attack_name,
            'Success on Source (%)': source_rate,
            'Success on Target (%)': target_rate,
            'Transfer Rate (%)': transfer_rate
        })

    df = pd.DataFrame(transfer_data)
    print("\nTransferability Results:")
    print(df.to_string(index=False))

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df['Success on Source (%)'], width, label='Source Model',
           color='steelblue')
    ax.bar(x, df['Success on Target (%)'], width, label='Target Model',
           color='coral')
    ax.bar(x + width, df['Transfer Rate (%)'], width, label='Transfer Rate',
           color='lightgreen')

    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Attack Transferability Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Attack'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/transferability_analysis.png', dpi=300)

    # Interpretation
    uap_transfer = df[df['Attack'] == 'UAP']['Transfer Rate (%)'].values[0]
    fgsm_transfer = df[df['Attack'] == 'FGSM']['Transfer Rate (%)'].values[0]
    pgd_transfer = df[df['Attack'] == 'PGD']['Transfer Rate (%)'].values[0]

    interpretation = f"""
    Transferability Analysis Results:

    Transfer Rates:
    - UAP: {uap_transfer:.1f}%
    - FGSM: {fgsm_transfer:.1f}%
    - PGD: {pgd_transfer:.1f}%

    Hypothesis Validation:
    {"✓ CONFIRMED: UAP transfers better than per-image attacks." if uap_transfer > max(fgsm_transfer, pgd_transfer) else
     "✗ REJECTED: UAP does not transfer better than per-image attacks."}

    Explanation:
    {"UAP exploits shared geometric properties of decision boundaries that are " +
     "consistent across architectures. Per-image attacks (FGSM/PGD) optimize for " +
     "the specific loss landscape of the source model, leading to overfitting " +
     "and poor transferability." if uap_transfer > max(fgsm_transfer, pgd_transfer) else
     "Unexpected result - may indicate that the two models have very different " +
     "decision boundaries, or that per-image attacks happened to find more " +
     "transferable directions by chance."}
    """

    print(interpretation)

    return {
        'dataframe': df,
        'interpretation': interpretation
    }
```

### Template 4: Complete Analysis Pipeline

```python
def run_complete_analysis(source_model, target_model, train_data, test_data, uap):
    """
    Run all required analysis experiments and generate report section
    """
    print("="*80)
    print("ANALYSIS & INTUITION - EXPERIMENT PIPELINE")
    print("="*80)

    # Question 1: Geometric Intuition
    print("\n" + "="*80)
    print("QUESTION 1: GEOMETRIC INTUITION")
    print("="*80)

    print("\n[1/3] Computing gradient correlation analysis...")
    correlation_results = analyze_shared_vulnerabilities(source_model, train_data)

    print("\n[2/3] Performing dimensionality analysis...")
    dimensionality_results = analyze_decision_boundary_dimensionality(
        source_model, train_data
    )

    print("\n[3/3] Analyzing UAP alignment with principal components...")
    # Check if UAP aligns with principal components
    uap_flat = uap.cpu().flatten().numpy()
    pca = dimensionality_results['pca']
    projections = np.abs(pca.transform([uap_flat])[0, :10])

    print(f"UAP projection on top 10 PCs: {projections}")
    print(f"Sum of top 10 projections: {projections.sum():.3f}")

    # Question 2: Comparative Analysis
    print("\n" + "="*80)
    print("QUESTION 2: COMPARATIVE ANALYSIS")
    print("="*80)

    print("\n[1/1] Testing transferability across architectures...")
    transferability_results = analyze_transferability(
        source_model, target_model, test_data, uap
    )

    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY FOR REPORT")
    print("="*80)

    summary = f"""
    ANALYSIS & INTUITION SUMMARY
    ============================

    Question 1: Why Does a Single Vector Fool Majority of Images?

    1. Shared Subspaces Evidence:
       {correlation_results['interpretation']}

    2. Dimensionality Contribution:
       {dimensionality_results['interpretation']}

    3. UAP Geometric Alignment:
       - UAP aligns with top principal components
       - Projection strength: {projections[:3].sum():.3f}
       - This confirms UAP exploits low-dimensional vulnerability structure

    Question 2: Comparative Analysis

    1. Transferability Results:
       {transferability_results['interpretation']}

    2. Vulnerability Exploited:
       - UAP: Exploits correlated decision boundary normals
       - FGSM: Exploits local gradient direction
       - PGD: Exploits optimal per-image perturbation

    3. Key Insight:
       Universal perturbations work because decision boundaries share
       geometric structure across the dataset, which is also shared
       across different model architectures.

    Figures Generated:
    - results/gradient_correlation.png
    - results/pca_dimensionality_analysis.png
    - results/transferability_analysis.png
    """

    print(summary)

    # Save summary to file
    with open('results/analysis_summary.txt', 'w') as f:
        f.write(summary)

    print("\n✓ Analysis complete! Use summary for report writing.")

    return {
        'correlation': correlation_results,
        'dimensionality': dimensionality_results,
        'transferability': transferability_results,
        'summary': summary
    }
```

## Key Challenges & Solutions

### Challenge 1: No Pre-trained Model
**Solution:**
- Train ResNet-18 on CIFAR-100 (85%+ accuracy achievable)
- Or use transfer learning from ImageNet

### Challenge 2: Slow Computation
**Solution:**
- Use GPU
- Reduce subset size initially (test with 100 images)
- Reduce max_iter_df

### Challenge 3: Low Fooling Rate
**Solution:**
- Increase ξ (perturbation budget)
- Increase max_iter_uni
- Ensure model is properly trained
- Check data normalization

### Challenge 4: Memory Issues
**Solution:**
- Process images sequentially in UAP loop
- Clear GPU cache periodically
- Use smaller batch sizes

## Code Quality Checklist

- [ ] Modular code structure
- [ ] Proper documentation and comments
- [ ] Type hints for functions
- [ ] Error handling
- [ ] Logging for debugging
- [ ] Reproducible (set random seeds)
- [ ] Efficient (GPU utilization)
- [ ] Visualizations are clear and labeled

## References

1. Moosavi-Dezfooli, S. M., Fawzi, A., Fawzi, O., & Frossard, P. (2017).
   "Universal adversarial perturbations."
   CVPR 2017.

2. Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016).
   "DeepFool: a simple and accurate method to fool deep neural networks."
   CVPR 2016.

## Submission Checklist

- [ ] Complete source code
- [ ] Pre-trained model weights
- [ ] Generated universal perturbation (v.pt)
- [ ] Fooling rate results
- [ ] All visualizations
- [ ] Analysis document/report
- [ ] README with instructions to reproduce

## Expected Deliverables

### 1. Implementation (25 Marks)

**Code Requirements:**
- [ ] Clean, modular, documented implementation
- [ ] DeepFool algorithm (correct implementation)
- [ ] UAP algorithm (follows Moosavi-Dezfooli et al. 2017)
- [ ] Projection function (L∞ or L2)
- [ ] Training on subset (500-1000 images)
- [ ] Convergence tracking and checkpointing

**Technical Requirements:**
- [ ] Uses pre-trained ResNet-18 or VGG-16 on CIFAR-100
- [ ] Perturbation magnitude constraint enforced
- [ ] GPU acceleration utilized
- [ ] Reproducible (random seeds set)

### 2. Evaluation (10 Marks)

**Quantitative Results:**
- [ ] Fooling rate ≥ 70% on CIFAR-100 test set
- [ ] Report perturbation magnitude (L∞ and L2 norms)
- [ ] Model accuracy comparison (clean vs perturbed)
- [ ] Convergence plot (fooling rate vs iteration)

**Visualizations Required:**
- [ ] Universal perturbation v (3-channel visualization)
- [ ] Example images: original vs perturbed (5-10 examples)
- [ ] Predictions before/after perturbation
- [ ] Analysis: Does perturbation show structural patterns or random noise?
  - Spatial visualization
  - Frequency analysis (FFT/DCT)
  - Per-channel analysis

**Comparison with FGSM and PGD:**
- [ ] Implement FGSM attack
- [ ] Implement PGD attack
- [ ] Comparison table with metrics:
  - Success rate
  - Average perturbation norm
  - Computation time
  - Visual quality (optional: SSIM/PSNR)

### 3. Analysis & Intuition (Critical for High Marks)

**Question 1: Geometric Intuition (5-7 marks expected)**

Must include:
- [ ] **Explanation:** Why does single vector fool majority of images?
- [ ] **Shared Subspaces:** Explain decision boundary correlation
- [ ] **Dimensionality Analysis:** Role of high-dimensional input space
- [ ] **Experimental Evidence:**
  - Gradient correlation matrix
  - PCA analysis of decision boundaries
  - Visualization of shared vulnerabilities
- [ ] **Mathematical Justification:**
  - Discuss effective dimensionality
  - Explain correlation of normal vectors
  - Connect to probability of universal directions

**Question 2: Comparative Analysis (5-7 marks expected)**

Must include:
- [ ] **Vulnerability Comparison:**
  - UAP vs FGSM vs PGD geometry
  - What each attack exploits
  - Why UAP is universal while others are per-image
- [ ] **Transferability Hypothesis:**
  - Clear hypothesis statement
  - Experimental validation (test on different architecture)
  - Transfer rate measurements
  - Justification based on geometric properties
- [ ] **Discussion:**
  - Why UAP might transfer better
  - Role of shared geometric properties
  - Implications for robustness

### 4. Report Structure

**Recommended Sections:**

1. **Introduction**
   - Background on adversarial perturbations
   - UAP motivation and significance
   - Assignment objectives

2. **Methodology**
   - DeepFool algorithm explanation
   - UAP algorithm explanation
   - Implementation details
   - Hyperparameter choices

3. **Experimental Setup**
   - Dataset: CIFAR-100
   - Model: ResNet-18/VGG-16
   - Training configuration
   - Hardware used

4. **Results**
   - Fooling rate on test set
   - Convergence analysis
   - Perturbation visualization
   - Example images

5. **Analysis & Intuition** ⭐ CRITICAL SECTION ⭐
   - Question 1: Geometric intuition
     - Shared subspaces explanation
     - Dimensionality analysis
     - Experimental evidence
   - Question 2: Comparative analysis
     - UAP vs FGSM vs PGD
     - Transferability experiments
     - Hypothesis validation

6. **Discussion**
   - Patterns in perturbation (structure vs noise)
   - Implications for model robustness
   - Limitations of approach
   - Potential defenses

7. **Conclusion**
   - Summary of findings
   - Key insights
   - Future work

8. **References**
   - Moosavi-Dezfooli et al. (2017)
   - Moosavi-Dezfooli et al. (2016) - DeepFool
   - Any other papers cited

### 5. Submission Package

**Files to submit:**
```
submission/
├── src/
│   ├── deepfool.py
│   ├── uap.py
│   ├── fgsm.py
│   ├── pgd.py
│   ├── evaluation.py
│   └── models.py
├── results/
│   ├── uap_perturbation.pt              # Saved universal perturbation
│   ├── uap_visualization.png            # Visualization of v
│   ├── convergence_plot.png             # Fooling rate vs iteration
│   ├── examples/                        # Original vs perturbed images
│   ├── gradient_correlation.png         # For analysis Q1
│   ├── pca_analysis.png                 # For analysis Q1
│   ├── comparison_table.png             # UAP vs FGSM vs PGD
│   └── transferability_results.png      # For analysis Q2
├── checkpoints/
│   └── resnet18_cifar100.pth            # Pre-trained model
├── report.pdf                           # Main report document
├── main.py                              # Main execution script
├── requirements.txt
└── README.md                            # Instructions to reproduce
```

**README Requirements:**
- Instructions to setup environment
- How to train/download model
- How to reproduce UAP computation
- How to reproduce all experiments
- How to generate all figures
- Expected runtime and hardware requirements