# Architecture Specification for TinyKAN-HAR Reproduction

## 1. TinyKAN-HAR Architecture

### 1.1 Overview

TinyKAN-HAR consists of three components:
1. **KAN Feature Extractor**: L-layer KAN that maps input windows to a d_L-dimensional latent vector z
2. **Classification Head**: Linear layer V + bias c mapping z → R^{C_s} logits for seen-class classification
3. **Zero-Shot Learning Module**: Semantic projection W_sem mapping z → R^m, cosine compatibility with class semantic embeddings S

### 1.2 KAN Layer Definition

Each KAN layer l (l = 1 … L) consists of:
- **Linear mixing**: u^{(l)} = W^{(l)} x^{(l-1)} + b^{(l)}, where W^{(l)} ∈ R^{d_l × d_{l-1}}, b^{(l)} ∈ R^{d_l}
- **Univariate spline transformation**: x_j^{(l)} = φ_j^{(l)}(u_j^{(l)}) where φ_j^{(l)}(u) = Σ_{k=1}^{K^{(l)}} θ_{j,k}^{(l)} B_k^{(l)}(u)
- **Dropout**: x^{(l)} = m^{(l)} ⊙ x^{(l)} with dropout probability p_drop

Output of layer l: x^{(l)} ∈ R^{d_l}
Final latent vector: z = x^{(L)} ∈ R^{d_L}

### 1.3 Configuration Used in Full TinyKAN-HAR

```
L = 3          # number of KAN layers
d_0 = T × D    # input dimension (window length × channels)
d_1 = 256      # layer 1 width
d_2 = 128      # layer 2 width
d_3 = d_L = 128  # latent dimension
K = 5          # number of B-spline basis functions per neuron
p_drop = 0.3   # dropout probability
```

### 1.4 Input Dimensions by Dataset

| Dataset | T (window length) | D (channels) | d_0 = T×D |
|---------|-------------------|-------------|-----------|
| UCI HAR | 128 | 6 | 768 |
| WISDM | 200 | 12 | 2400 |
| PAMAP2 | 250 | 28 | 7000 |

### 1.5 Spline Basis Functions

- **Type**: Cubic B-splines with uniform knots
- **Knots**: Cover the range [u_min, u_max] with K+1 interior knots (K+4 total for cubic)
- **Initialization**: Spline coefficients initialized to approximate identity function φ(u) ≈ u over the pre-defined input range
- **Regularization**: Smoothness penalty on discrete second differences of θ coefficients (λ_smooth = 0.01)

### 1.6 Semantic Embedding Configuration

- **Hybrid representation**: Concatenation of attribute vectors (binary attributes per activity) and text embeddings from sentence-transformers
- **Text embedding model**: `all-MiniLM-L6-v2` (384-dim output)
- **Attribute dimension**: 8 binary attributes per activity class
- **Final hybrid dimension**: m = 8 + 384 = 392
- **Projection**: W_sem ∈ R^{392 × 128}
- **Temperature τ**: 0.07

### 1.7 Classification Head

- **Weight matrix**: V ∈ R^{C_s × 128}
- **Bias vector**: c ∈ R^{C_s}
- **C_s (seen classes)**: UCI HAR = 4, WISDM = 14, PAMAP2 = 14

### 1.8 ZSL Calibration

- **Calibration factor**: γ = 0.5 (selected on validation set)
- **Calibrated score**: ḡ(z, s_y) = g(z, s_y) - γ if y ∈ Y_s, else g(z, s_y)

## 2. Baseline Models

### 2.1 Classical ML Baselines

All classical baselines operate on hand-crafted feature vectors extracted per window.

**Feature extraction** g_feat(·):
- Time domain: mean, std, RMS, max, min, zero-crossing rate, signal magnitude area
- Frequency domain: dominant frequency, spectral entropy, energy in bands
- Produces feature vector of dimension ~50-100 per channel

**kNN**:
- k = 5 (selected on validation)
- Euclidean distance
- Majority vote

**SVM**:
- RBF kernel
- C = 10.0, γ = 0.01 (selected via cross-validation)
- One-vs-rest multiclass

**Random Forest**:
- 100 trees
- Max depth = 15
- Bootstrap sampling with replacement

### 2.2 Deep Learning Baselines

All deep baselines take normalized window tensor X̂ ∈ R^{T×D} as input.

**1D-CNN**:
- Conv1D(64, kernel=7, stride=1) → BatchNorm → ReLU
- Conv1D(128, kernel=5, stride=1) → BatchNorm → ReLU
- MaxPool1D(pool_size=2)
- Conv1D(256, kernel=3, stride=1) → BatchNorm → ReLU
- GlobalAveragePooling1D
- Dense(128) → ReLU → Dropout(0.3)
- Dense(C_s)

**LSTM**:
- Input: X̂ reshaped to (T, D)
- LSTM(128, return_sequences=True)
- LSTM(64)
- Flatten or take last hidden state
- Dense(128) → ReLU → Dropout(0.3)
- Dense(C_s)

**CNN-LSTM**:
- CNN front-end: Conv1D(64) + Conv1D(128) + MaxPool1D
- LSTM(64, return_sequences=False)
- Dense(128) → ReLU → Dropout(0.3)
- Dense(C_s)

**Transformer**:
- Input: X̂ reshaped to (T, D)
- Linear projection to d_model = 64
- Positional encoding (sinusoidal)
- 2 Transformer encoder layers, 4 heads each, d_ff = 128
- GlobalAveragePooling
- Dense(C_s)

### 2.3 ZSL Baselines (for Experiment 2)

Each baseline uses the same semantic projection and cosine compatibility as TinyKAN-HAR, but replaces the KAN feature extractor with:

- **CNN+ZSL**: 1D-CNN backbone (same as above), last hidden layer → W_sem → cosine compatibility
- **LSTM+ZSL**: LSTM backbone (same as above), last hidden state → W_sem → cosine compatibility
- **Transformer+ZSL**: Transformer encoder (same as above), pooled output → W_sem → cosine compatibility

## 3. Training Configuration

### 3.1 Optimizer

- **Algorithm**: Adam
- **Initial learning rate**: η_0 = 1e-3
- **Schedule**: Cosine annealing from η_0 to 1e-5 over E_max epochs
- **Batch size**: N_b = 64

### 3.2 Loss Functions

**Task loss for seen-class classification**:
L_CE = - (1/N_b) Σ log p_{i,y_i}  (cross-entropy)

**ZSL alignment loss**:
L_align = (1/N_b) Σ ||h_i - s_{y_i}||_2^2

**ZSL semantic cross-entropy**:
L_sem-CE = - (1/N_b) Σ log q_{i,y_i}

**Combined ZSL loss**:
L_ZSL = λ_align * L_align + λ_sem-CE * L_sem-CE

**Overall task loss**:
L_task = L_CE + λ_ZSL * L_ZSL

**Total loss with regularization**:
L_total = L_task + λ_lin * R_lin + λ_smooth * R_smooth

### 3.3 Hyperparameters

| Parameter | Value |
|-----------|-------|
| λ_ZSL | 1.0 |
| λ_align | 0.5 |
| λ_sem-CE | 0.5 |
| λ_lin (weight decay) | 1e-4 |
| λ_smooth | 0.01 |
| p_drop | 0.3 |
| E_max | 150 |
| Patience | 20 epochs |
| Seed | 42 (primary), 0, 7, 13, 21 (for multi-seed exp) |

### 3.4 Initialization

- Linear layers (W, b): Xavier initialization
- Spline coefficients: Least-squares fit to identity over predefined grid
- W_sem: Near-zero initialization (scale = 0.01)

## 4. Dataset Splits

### 4.1 UCI HAR

| Split | Subjects | Windows (approx.) |
|-------|----------|-------------------|
| Train | 17 | ~5500 |
| Validation | 4 | ~1300 |
| Test | 9 | ~2900 |

**Seen classes**: WALKING, SITTING, STANDING, LAYING (4)
**Unseen classes**: WALKING_UPSTAIRS, WALKING_DOWNSTAIRS (2)

### 4.2 WISDM

| Split | Subjects |
|-------|----------|
| Train | 35 |
| Validation | 8 |
| Test | 8 |

**Seen classes** (14): walking, jogging, sitting, standing, climbing stairs, drinking, brushing teeth, etc.
**Unseen classes** (4): jogging, walking_upstairs, walking_downstairs, jumping

### 4.3 PAMAP2

| Split | Subjects |
|-------|----------|
| Train | 6 |
| Validation | 1 |
| Test | 2 |

**Seen classes** (14): lying, sitting, standing, walking, etc.
**Unseen classes** (4): running, rope_jumping, vacuum_cleaning, ironing

## 5. TinyML Deployment Architecture

### 5.1 Quantization Configuration

- **Target precision**: 8-bit uniform symmetric quantization
- **Quantization scheme**: Post-training quantization with optional fine-tuning
- **Activation scaling**: Per-layer scale factors stored in float
- **Weight storage**: int8 weights + float scale factors

### 5.2 LUT-Based Spline Evaluation

For each neuron, the spline function φ_j(u) is pre-sampled on a uniform grid of N_grid = 64 points over the input range [u_min, u_max]. Evaluation at runtime uses linear interpolation between table entries.

### 5.3 Memory Budget

Target MCU: Cortex-M4F @ 80MHz, 256 kB flash, 64 kB SRAM

| Component | Memory |
|-----------|--------|
| Quantized weights (int8) | ~140 kB |
| LUT entries (int8 + scales) | ~5 kB |
| Activation buffers (int8) | ~26 kB |
| Total | ~145 kB flash, ~26 kB RAM |

## 6. Evaluation Metrics

### 6.1 HAR Metrics

- **Overall accuracy**: (TP + TN) / total
- **Macro-F1**: (1/C) Σ F1_c for each class c
- **Micro-F1**: 2·(precision·recall)/(precision+recall) aggregated over all classes

### 6.2 ZSL Metrics

- **Acc_ZSL**: Accuracy on unseen classes only (pure ZSL)
- **Acc_seen**: Accuracy on seen classes in gZSL setting
- **Acc_unseen**: Accuracy on unseen classes in gZSL setting
- **H**: Harmonic mean = 2·Acc_seen·Acc_unseen / (Acc_seen + Acc_unseen)

### 6.3 TinyML Metrics

- **Model size**: Flash footprint in kB
- **Peak RAM**: Maximum SRAM usage during inference in kB
- **Latency**: Average wall-clock time per window in ms
- **Energy**: Estimated energy per inference in µJ

## 7. Experiment Matrix

| Exp | Configuration | Variants |
|-----|---------------|----------|
| 1 | Full TinyKAN-HAR, L=3, d_L=128 | All 8 models × 3 datasets |
| 2 | TinyKAN-HAR + baselines | 4 models × 2 datasets (UCI, PAMAP2) |
| 3 | TinyKAN-HAR fixed, γ varies | γ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} |
| 4 | TinyKAN-HAR + Transformer+ZSL | 2 models × 2 datasets × 5 seeds |
| 5 | Full TinyKAN-HAR | 20 ablation variants on UCI HAR |
| 6 | All int8 models | 4 models, measured on target MCU |

## 8. Code Structure

```
src/
├── data/
│   ├── datasets.py         # UCI HAR, WISDM, PAMAP2 loaders
│   ├── preprocessing.py    # Gravity separation, normalization, windowing
│   └── splits.py           # Seen/unseen splits and subject-based folds
├── models/
│   ├── kan.py              # KAN layer, spline functions
│   ├── tinykan_har.py      # Full TinyKAN-HAR model
│   ├── baselines.py        # 1D-CNN, LSTM, CNN-LSTM, Transformer
│   └── classical.py        # kNN, SVM, RF with feature extraction
├── training/
│   ├── trainer.py          # Training loop with early stopping
│   ├── losses.py           # CE, alignment, semantic CE losses
│   └── optimization.py     # Adam optimizer, cosine schedule
├── zsl/
│   ├── semantic_embeddings.py  # Text + attribute embeddings
│   └── compatibility.py    # Cosine compatibility, calibration
├── explainability/
│   ├── attribution.py      # Gradient-based attributions
│   ├── shap.py             # SHAP-style feature importance
│   └── visualization.py    # Spline plotting, heatmaps
├── tinyml/
│   ├── quantization.py     # PTQ and QAT
│   ├── lut.py              # LUT generation for splines
│   └── export.py           # TFLite Micro exporter
└── evaluation/
    ├── metrics.py          # HAR, ZSL, TinyML metrics
    └── tables.py           # Results table generation
```

## 9. Expected Results Summary

### Table 3 Targets (Experiment 1)

| Model | UCI Acc | UCI F1 | WISDM Acc | WISDM F1 | PAMAP2 Acc | PAMAP2 F1 |
|-------|---------|--------|-----------|---------|-----------|---------|
| kNN | 96.2 | 96.0 | 96.1 | 95.8 | 96.0 | 95.7 |
| SVM | 96.8 | 96.5 | 96.5 | 96.2 | 96.3 | 96.0 |
| RF | 97.0 | 96.7 | 96.9 | 96.6 | 96.5 | 96.2 |
| 1D-CNN | 97.6 | 97.3 | 97.2 | 97.0 | 96.9 | 96.7 |
| LSTM | 97.1 | 96.9 | 96.8 | 96.6 | 96.4 | 96.1 |
| CNN-LSTM | 97.8 | 97.5 | 97.4 | 97.1 | 97.0 | 96.8 |
| Transformer | 98.0 | 97.7 | 97.9 | 97.7 | 97.3 | 97.1 |
| TinyKAN-HAR | 98.3 | 98.0 | 97.9 | 97.7 | 97.3 | 97.1 |

### Table 4 Targets (Experiment 2)

| Model | UCI Acc_ZSL | UCI Acc_seen | UCI Acc_unseen | UCI H | PAMAP2 Acc_ZSL | PAMAP2 Acc_seen | PAMAP2 Acc_unseen | PAMAP2 H |
|-------|------------|-------------|---------------|-------|---------------|---------------|------------------|---------|
| CNN+ZSL | 91.8 | 91.8 | 88.5 | 92.6 | 90.9 | 96.3 | 87.2 | 91.4 |
| LSTM+ZSL | 90.7 | 90.7 | 87.1 | 91.7 | 89.8 | 96.0 | 86.0 | 90.7 |
| Transformer+ZSL | 93.2 | 97.4 | 90.1 | 93.7 | 92.0 | 96.6 | 89.2 | 92.8 |
| TinyKAN-HAR | 96.4 | 98.1 | 95.0 | 96.7 | 96.0 | 97.5 | 94.6 | 96.0 |

### Table 6 Targets (Experiment 4)

| Model | Acc (%) | H (%) | p-value |
|-------|---------|-------|---------|
| Transformer+ZSL | 97.7±0.2 | 93.3±0.4 | - |
| TinyKAN-HAR | 98.3±0.1 | 96.4±0.3 | <0.01 |

### Table 10 Targets (Experiment 6)

| Model | Acc (%) | Model (kB) | RAM (kB) | Latency (ms) | Energy (µJ) |
|-------|---------|-----------|---------|------------|------------|
| 1D-CNN Tiny | 97.4 | 120 | 24 | 3.5 | 280 |
| CNN-LSTM Tiny | 97.6 | 165 | 30 | 6.1 | 450 |
| Transformer Tiny | 97.9 | 210 | 36 | 7.8 | 520 |
| TinyKAN-HAR Tiny | 98.3 | 145 | 26 | 4.1 | 320 |