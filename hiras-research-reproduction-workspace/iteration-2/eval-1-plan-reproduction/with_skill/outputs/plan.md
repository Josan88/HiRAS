# Phase 1: Overall Planning — TinyKAN-HAR Reproduction

## Paper Summary
**Title:** Explainable Kolmogorov-Arnold Networks for Zero-Shot Human Activity Recognition on TinyML Edge Devices  
**Authors:** Lamaakal et al. (2026)  
**Venue:** Sensors (MDPI, Open Access)  
**Core Claim:** TinyKAN-HAR achieves >97% macro-F1 on seen classes, >96% ZSL accuracy on unseen classes, harmonic mean >96% in gZSL, with only 145 kB flash and 4.1 ms latency on Cortex-M4F.

---

## Phase 0: Intake Summary

### Core Contributions
1. Compact KAN-based feature extractor with learnable 1D spline nonlinearities
2. Semantic-embedding-based ZSL module enabling pure and generalized zero-shot HAR
3. Multi-level explainability (gradient attributions, SHAP, spline visualization)
4. TinyML deployment with 8-bit quantization, LUT-based spline evaluation, and structured pruning

### Datasets
| Dataset | #Subjects | #Activities | Fs (Hz) | Sensors | Seen | Unseen |
|---------|----------|-------------|---------|---------|------|--------|
| UCI HAR | 30 | 6 | 50 | Phone Acc+Gyro | 4 | 2 |
| WISDM | 51 | 18 | 20 | Phone+Watch | 14 | 4 |
| PAMAP2 | 9 | 18 | 100→50 | 3 IMUs + HR | 14 | 4 |

### Preprocessing
- Temporal alignment and resampling to 50 Hz target
- Gravity separation via low-pass filter (0.3–0.5 Hz cutoff)
- Per-channel z-score normalization (computed on training only)
- Fixed-length windows: UCI HAR T=128, WISDM T=200, PAMAP2 T=250
- 50% overlap, majority-vote labeling
- Missing value interpolation (PAMAP2)

### Hyperparameters (stated in paper)
- Optimizer: Adam
- Initial learning rate: 10^-4 to 10^-3 (range stated, exact value not specified)
- Batch size: 32, 64, 128 (choices mentioned)
- Max epochs: 100–200
- L2 weight decay, smoothness regularization, dropout (exact values not specified)
- ZSL loss weights: λ_ZSL, λ_align, λ_sem-CE (exact values not specified)
- Calibration factor: γ = 0.5 (selected on validation)
- KAN depth: L = 3, latent dim d_L = 128 (selected as best trade-off)
- Spline basis: cubic B-splines with learnable coefficients
- KAN initialization: splines initialized to approximate identity function

### Model Architecture
- KAN with L=3 layers, pyramidal widths (d_0 > d_1 > d_2 > d_L=128)
- Linear mixing W(l) + bias b(l) followed by univariate spline φ_j(u)
- Classification head: linear V + bias c over seen classes
- Semantic projection: W_sem ∈ R^{m × d_L} mapping latent to semantic space
- Compatibility: cosine similarity between projected h and semantic embedding s_y
- Score calibration: subtract γ from seen-class scores in gZSL

### Training Losses
- Cross-entropy L_CE over seen classes
- Semantic alignment L_align (L2 distance between h and s_y)
- Semantic softmax cross-entropy L_sem-CE over seen compatibility scores
- Total: L_task = L_CE + λ_ZSL(λ_align*L_align + λ_sem-CE*L_sem-CE)
- Regularization: L_lin (weight decay) + L_smooth (spline second differences)

### Ambiguities / Missing Information
- Exact learning rate value (only range given)
- Exact batch size chosen (multiple options listed)
- Exact λ_ZSL, λ_align, λ_sem-CE values
- Exact dropout probability
- Exact number of spline knots/basis functions K
- Exact initialization range for linear weights
- Whether ZSL experiments reported in Table 4 include WISDM (not shown)
- Exact semantic embedding dimension m
- Whether hybrid embeddings use concatenation or averaging

---

## Experiment Plan

### Experiment 1: HAR Performance on Seen Classes (Section 5.1, Table 3)
**Goal:** Validate that TinyKAN-HAR achieves competitive accuracy and macro-F1 on seen classes across all three datasets.

**Sub-experiments:**
- 1a: UCI HAR — TinyKAN-HAR vs 7 baselines (kNN, SVM, RF, 1D-CNN, LSTM, CNN-LSTM, Transformer)
- 1b: WISDM — Same comparison on 14 seen classes
- 1c: PAMAP2 — Same comparison on 14 seen classes

**Metrics:** Overall Accuracy (%), Macro-F1 (%)

**Dependencies:** Data download and preprocessing (Experiments 0a–0c)

**Expected Outcome:** TinyKAN-HAR achieves 98.3% Acc / 98.0% F1 on UCI HAR, 97.9% Acc / 97.7% F1 on WISDM, 97.3% Acc / 97.1% F1 on PAMAP2

---

### Experiment 2: Zero-Shot and Generalized Zero-Shot Performance (Section 5.2, Table 4)
**Goal:** Validate that TinyKAN-HAR's ZSL module enables recognition of unseen activities via semantic embeddings.

**Sub-experiments:**
- 2a: UCI HAR — Pure ZSL (Acc_ZSL) and Generalized ZSL (Acc_seen, Acc_unseen, H) vs CNN+ZSL, LSTM+ZSL, Transformer+ZSL
- 2b: PAMAP2 — Same comparison (WISDM ZSL results not reported in Table 4)

**Metrics:** Pure ZSL Accuracy Acc_ZSL (%), Generalized ZSL Acc_seen (%), Acc_unseen (%), Harmonic Mean H (%)

**Dependencies:** Data preprocessing, KAN-HAR training with ZSL losses

**Expected Outcome:** UCI HAR: Acc_ZSL=96.4%, Acc_seen=98.1%, Acc_unseen=95.0%, H=96.7%. PAMAP2: Acc_ZSL=96.0%, Acc_seen=97.5%, Acc_unseen=94.6%, H=96.0%

---

### Experiment 3: Robustness of Calibration Factor γ (Section 5.3, Table 5, Figure 2)
**Goal:** Verify that reported ZSL performance is robust to γ choice and not overfitted to a specific value.

**Sub-experiments:**
- 3a: Sweep γ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} on UCI HAR
- 3b: Evaluate on both validation and test splits
- 3c: Plot Acc_ZSL and H vs γ (reproduce Figure 2)

**Metrics:** Acc_ZSL, Acc_seen, Acc_unseen, H for each γ value

**Dependencies:** Trained KAN-HAR model, ZSL evaluation pipeline

**Expected Outcome:** γ=0.5 is optimal on validation, H remains >96% for γ ∈ [0.25, 1.0]

---

### Experiment 4: Statistical Significance Across Random Seeds (Section 5.4, Table 6)
**Goal:** Assess robustness of gains and provide statistical significance via multi-seed evaluation.

**Sub-experiments:**
- 4a: Train 5 runs of KAN-HAR with different random seeds on UCI HAR and PAMAP2
- 4b: Train 5 runs of Transformer+ZSL baseline with same seeds
- 4c: Compute mean ± std of Acc and H across seeds
- 4d: Paired two-sided t-test on H values between KAN-HAR and Transformer+ZSL

**Metrics:** Mean Acc (%), Mean H (%), p-value on H

**Dependencies:** Multi-run training pipeline, statistical testing

**Expected Outcome:** KAN-HAR: Acc=98.3±0.1%, H=96.4±0.3%. Transformer+ZSL: Acc=97.7±0.2%, H=93.3±0.4%. p < 0.01

---

### Experiment 5: Ablation Studies (Section 5.7, Tables 7, 8, 9)

#### 5A: Component Ablation on UCI HAR (Table 7)
**Goal:** Quantify contribution of each architectural and deployment choice.

**20 Variants to test:**
1. Full TinyKAN-HAR (int8, L=3, d_L=128) — baseline
2. Full TinyKAN-HAR (FP32, L=3, d_L=128)
3. w/o ZSL losses
4. w/o calibrated scores
5. w/o semantic projection layer
6. w/o explainability regularizer
7. Shallow KAN (L=1, d_L=64)
8. Deep KAN (L=4, d_L=128)
9. Narrow latent (L=3, d_L=64)
10. Wide latent (L=3, d_L=256)
11. Coarse spline (fewer knots)
12. Fine spline (more knots)
13. w/o LUTs (direct spline evaluation)
14. Quantization only (no LUT)
15. Quant. + 50% structured pruning
16. Quant. + 70% structured pruning
17. Short window (reduced T)
18. Long window (increased T)
19. Low dropout (p=0.1)
20. High dropout (p=0.5)

**Metrics (all variants):** Acc, F1_macro, Acc_ZSL, H, Model [kB], RAM [kB], Latency [ms], Energy [µJ]

**Dependencies:** Model training with specific configurations, TinyML profiling

#### 5B: Hybrid Semantic Embeddings (Table 8)
**Goal:** Assess sensitivity to semantic representation choice.

**3 Variants:**
- ATTR: Manually defined attributes only
- TEXT: Text embeddings from pretrained language model only
- HYBRID: Concatenation of attributes + text embeddings + learned projection

**Metrics:** Acc_ZSL, H on UCI HAR and PAMAP2

#### 5C: Role of Semantic Structure (Table 9)
**Goal:** Verify ZSL behavior is driven by genuine semantics, not incidental structure.

**3 Variants:**
- HYBRID: Meaningful embeddings (baseline)
- RANDOM: Gaussian random embeddings
- SHUFFLED: Hybrid geometry but permuted label alignment

**Metrics:** Seen Acc, Acc_ZSL, H

**Dependencies:** Semantic embedding generation, controlled training

---

### Experiment 6: TinyML Deployment Metrics (Section 5.8, Table 10)
**Goal:** Validate that TinyKAN-HAR fits within MCU constraints while maintaining accuracy.

**Sub-experiments:**
- 6a: Profile int8 TinyKAN-HAR on Cortex-M4F (256kB flash, 64kB SRAM, 80MHz)
- 6b: Profile int8 baselines (1D-CNN Tiny, CNN-LSTM Tiny, Transformer Tiny)
- 6c: Measure model size, peak RAM, latency, energy per inference

**Metrics:** Model [kB], Peak RAM [kB], Latency [ms], Energy [µJ]

**Dependencies:** Model quantization, TFLM conversion, MCU benchmarking infrastructure

**Expected Outcome:** TinyKAN-HAR: 145 kB flash, 26 kB RAM, 4.1 ms, 320 µJ. Best trade-off vs baselines.

---

### Experiment 7: Case Studies and Visualization (Section 5.5, Figures 3, 4)
**Goal:** Qualitatively validate explainability mechanisms.

**Sub-experiments:**
- 7a: Attribution heatmaps for walking, sitting, ascending stairs (UCI HAR)
- 7b: Sensor-level relevance bar plots per activity
- 7c: Temporal relevance curves showing gait cycles
- 7d: Learned univariate spline functions from different layers
- 7e: Class-wise mean activation plots for selected neurons

**Activities:** Walking, Sitting, Ascending stairs (and extended in PAMAP2: Lying, Running)

**Dependencies:** Trained KAN-HAR, attribution computation, spline extraction

---

### Experiment 8: Effect of KAN Depth and Latent Dimension (Section 5.6)
**Goal:** Analyze how L and d_L affect accuracy and resource usage.

**Sub-experiments:**
- 8a: L ∈ {1, 2, 3, 4} at d_L=128
- 8b: d_L ∈ {64, 128, 256} at L=3
- 8c: Validate L=3, d_L=128 is optimal trade-off

**Metrics:** Acc, F1, Acc_ZSL, H, Model [kB], Latency [ms]

**Dependencies:** Multi-configuration training

---

## Implementation Dependencies

```
Data Download & Preprocessing (EXPERIMENT 0)
    ├── UCI HAR download + preprocess → Experiment 1a, 2a, 3, 4a, 5A, 5B, 5C, 8
    ├── WISDM download + preprocess → Experiment 1b
    └── PAMAP2 download + preprocess → Experiment 1c, 2b, 4a, 5A, 5B, 5C

KAN-HAR Training (shared backbone for Experiments 1–8)
    ├── KAN layer implementation
    ├── Spline function parameterization
    ├── ZSL semantic module
    ├── Explainability hooks
    └── Quantization & LUT generation

Baseline Training (for Experiments 1, 2)
    ├── kNN, SVM, RF (classical)
    └── 1D-CNN, LSTM, CNN-LSTM, Transformer (deep)

TinyML Conversion (for Experiments 5A, 6)
    ├── 8-bit quantization
    ├── LUT generation for splines
    ├── TFLM conversion
    └── MCU profiling
```

---

## Count of Experiments

| # | Experiment | Paper Section | Tables/Figures |
|---|------------|---------------|----------------|
| 1 | HAR on seen classes | 5.1 | Table 3 |
| 2 | ZSL and gZSL | 5.2 | Table 4 |
| 3 | γ robustness | 5.3 | Table 5, Figure 2 |
| 4 | Statistical significance | 5.4 | Table 6 |
| 5 | Ablation studies | 5.7 | Tables 7, 8, 9 |
| 6 | TinyML deployment | 5.8 | Table 10 |
| 7 | Case study visualizations | 5.5 | Figures 3, 4 |
| 8 | Depth/width analysis | 5.6 | (narrative) |

**Total: 8 main experiment groups covering all 6 areas specified in the task**
