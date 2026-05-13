# Plan: TinyKAN-HAR — Explainable KAN for Zero-Shot HAR on TinyML

## Overview

This plan covers every experiment described in the paper "Explainable Kolmogorov–Arnold Networks for Zero-Shot Human Activity Recognition on TinyML Edge Devices" (Lamaakal et al., 2026). The paper proposes TinyKAN-HAR, a KAN-based architecture with a zero-shot learning module and multi-level explainability, deployed on Cortex-M4F MCUs.

---

## Experiment 1 — HAR Performance on Seen Classes (Section 5.1)

**Goal:** Establish that TinyKAN-HAR achieves competitive supervised HAR accuracy on seen classes across three datasets.

**Datasets:**
- UCI HAR (30 subjects, 6 activities, 50 Hz, phone Acc+Gyro, 4 seen / 2 unseen split)
- WISDM (51 subjects, 18 activities, 20 Hz, phone+watch, 14 seen / 4 unseen split)
- PAMAP2 (9 subjects, 18 activities, 100→50 Hz, 3 IMUs + HR, 14 seen / 4 unseen split)

**Models to implement/compare:**
- Classical baselines: kNN, SVM (RBF), Random Forest
- Deep baselines: 1D-CNN, LSTM, CNN-LSTM, Transformer
- Proposed: TinyKAN-HAR (L=3, d_L=128)

**Evaluation protocol:**
- Subject-disjoint train/val/test splits as described in Section 3.1
- Metrics: overall accuracy (%), macro-F1 (%)
- Train only on seen classes Y_s

**Expected outcomes:**
- TinyKAN-HAR achieves ≥97.3% accuracy and ≥97.1% macro-F1 across all three datasets
- TinyKAN-HAR outperforms all baselines on each dataset

**Dependencies:** None (base experiment)

---

## Experiment 2 — Zero-Shot and Generalized Zero-Shot Performance (Section 5.2)

**Goal:** Evaluate TinyKAN-HAR's ability to recognize unseen activities via semantic embeddings, in both pure ZSL and generalized ZSL (gZSL) settings.

**Datasets:** Same three datasets (UCI HAR, WISDM, PAMAP2) with their seen/unseen splits

**Models:**
- CNN + ZSL head
- LSTM + ZSL head
- Transformer + ZSL head
- TinyKAN-HAR (ours)

**Evaluation protocol:**
- Pure ZSL: test only on unseen classes Y_u, report Acc_ZSL (%)
- gZSL: test on both seen and unseen classes; report Acc_seen, Acc_unseen, harmonic mean H (%)

**Expected outcomes:**
- TinyKAN-HAR: Acc_ZSL ≥ 96.0% on UCI HAR, ≥ 96.0% on PAMAP2; H ≥ 96.0% on both
- Significantly outperforms all ZSL baselines

**Dependencies:** Experiment 1 (KAN architecture must exist before adding ZSL module)

---

## Experiment 3 — Robustness of Calibration Factor γ (Section 5.3)

**Goal:** Verify that the gZSL score calibration factor γ is not over-tuned.

**Evaluation protocol:**
- Fix trained TinyKAN-HAR model
- Sweep γ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
- Report Acc_ZSL, Acc_seen, Acc_unseen, H for each γ on both validation and test splits
- γ* = 0.5 selected on validation set, applied to test set unchanged

**Expected outcomes:**
- H remains above 96% for γ ∈ [0.25, 1.0]
- γ* = 0.5 generalizes from validation to test without degradation

**Dependencies:** Experiment 2 (trained TinyKAN-HAR model required)

---

## Experiment 4 — Statistical Significance Across Random Seeds (Section 5.4)

**Goal:** Confirm that TinyKAN-HAR's gains are robust and statistically significant.

**Evaluation protocol:**
- Train 5 runs (different random seeds) of TinyKAN-HAR and Transformer+ZSL on UCI HAR and PAMAP2
- Report mean ± std of accuracy and harmonic mean H
- Paired two-sided t-test on H values; p < 0.01 expected

**Expected outcomes:**
- TinyKAN-HAR: Acc ≥ 98.3 ± 0.1%, H ≥ 96.4 ± 0.3%
- Statistically significant improvement over Transformer+ZSL baseline

**Dependencies:** Experiments 1–2

---

## Experiment 5 — Case Studies and Visualization of Explanations (Section 5.5)

**Goal:** Demonstrate multi-level explainability on representative activities.

**Activities analyzed:** Walking, Sitting, Ascending stairs (UCI HAR)

**Explanation types produced:**
- Attribution heatmaps (T × D matrices) overlaid on sensor signals
- Sensor-level relevance bar plots (per-channel and per-device-group)
- Temporal relevance curves (per time-step)
- Learned univariate KAN spline functions ϕ_j^{(l)}(u) with class-wise activation profiles

**Expected outcomes:**
- Walking: high attributions on vertical accelerometer/gyroscope, periodic peaks in temporal relevance aligned with gait cycles
- Sitting: low-frequency accelerometer emphasis, flat temporal relevance
- Ascending stairs: strong attributions on lower-body sensors during impact phases

**Dependencies:** Experiments 1–2 (trained model required)

---

## Experiment 6 — Effect of KAN Depth and Latent Dimension (Section 5.6)

**Goal:** Determine optimal L and d_L trade-off between accuracy and resource usage.

**Configurations tested:**
- L = 1, 2, 3, 4 with d_L = 128
- d_L = 64, 128, 256 with L = 3

**Evaluation protocol:**
- Train and evaluate on UCI HAR
- Report accuracy and macro-F1
- Report TinyML metrics (model size, RAM, latency, energy)

**Expected outcomes:**
- L = 3, d_L = 128 selected as optimal trade-off
- Accuracy saturates beyond L = 3 and d_L = 128 with negligible gain but higher cost

**Dependencies:** Experiments 1–2

---

## Experiment 7 — Ablation Studies (Section 5.7, Table 7)

**Goal:** Quantify contribution of each architectural and deployment choice.

### 7a. Precision and Quantization (Rows 1–2 of Table 7)
- TinyKAN-HAR (int8, L=3, d_L=128) vs. TinyKAN-HAR (FP32, L=3, d_L=128)
- Metrics: Acc, F1 macro, Acc_ZSL, H, model size, RAM, latency, energy

### 7b. ZSL Loss and Calibration (Rows 3–4)
- w/o ZSL losses: remove λ_ZSL alignment and semantic-CE terms
- w/o calibrated scores: disable score calibration (γ = 0)

### 7c. Semantic Projection and Explainability (Rows 5–6)
- w/o semantic projection layer: replace learned W_sem with identity
- w/o explainability regularizer: remove smoothness regularization

### 7d. KAN Depth and Width (Rows 7–10)
- Shallow KAN (L=1, d_L=64)
- Deep KAN (L=4, d_L=128)
- Narrow latent (L=3, d_L=64)
- Wide latent (L=3, d_L=256)

### 7e. Spline Resolution (Rows 11–12)
- Coarse spline (fewer knots)
- Fine spline (more knots)

### 7f. LUT vs. Direct Spline Evaluation (Rows 13–14)
- w/o LUTs (direct spline evaluation)
- Quantization only (no LUT)

### 7g. Structured Pruning (Rows 15–16)
- Quant. + 50% structured pruning
- Quant. + 70% structured pruning

### 7h. Window Length (Rows 17–18)
- Short window (reduced T)
- Long window (increased T)

### 7i. Dropout Rate (Rows 19–20)
- Low dropout (p=0.1)
- High dropout (p=0.5)

**Expected outcomes:** Each ablation confirms that the full configuration (int8, L=3, d_L=128, with ZSL losses, calibration, LUTs) achieves the best trade-off; specific quantified contributions in Table 7.

**Dependencies:** Experiments 1–2

---

## Experiment 8 — Ablation on Semantic Representations (Section 5.7.1, Table 8)

**Goal:** Assess sensitivity to choice of semantic representation.

**Variants:**
- ATTR: manually defined attribute vectors only
- TEXT: pretrained textual embeddings only
- HYBRID: concatenation of attributes + text embeddings with learned projection

**Expected outcomes:**
- HYBRID achieves best performance: Acc_ZSL ≥ 96.8% on UCI HAR, ≥ 96.3% on PAMAP2; H ≥ 97.1% / 96.6%

**Dependencies:** Experiments 1–2 (ZSL module must exist)

---

## Experiment 9 — Isolating Semantic Information (Section 5.7.2, Table 9)

**Goal:** Verify that zero-shot behavior is driven by genuine semantic structure, not incidental geometry.

**Variants:**
- HYBRID: meaningful embeddings (baseline)
- RANDOM: random Gaussian embeddings per class
- SHUFFLED: meaningful embeddings randomly permuted across labels

**Expected outcomes:**
- Seen-class accuracy remains ~98% for all variants
- ZSL metrics collapse to ~20–25% Acc_ZSL and ~34–39% H for RANDOM/SHUFFLED
- Proves semantic structure is essential for ZSL

**Dependencies:** Experiments 1–2, 8

---

## Experiment 10 — TinyML Deployment on Microcontroller (Section 5.8, Table 10)

**Goal:** Benchmark all models on actual Cortex-M4F-class MCU.

**Target hardware:** Cortex-M4F-class MCU, 256 kB flash, 64 kB SRAM, 80 MHz

**Models (int8):**
- 1D-CNN Tiny
- CNN-LSTM Tiny
- Transformer Tiny
- TinyKAN-HAR Tiny (ours)

**Metrics:** Model size (kB), peak RAM (kB), latency (ms), energy (μJ)

**Expected outcomes:**
- TinyKAN-HAR Tiny: 145 kB flash, 26 kB RAM, 4.1 ms latency, 320 μJ energy
- Best accuracy (98.3%) among TinyML baselines with favorable resource trade-off

**Dependencies:** Experiments 1–2 (int8 quantization and model export)

---

## Experiment 11 — Qualitative Interpretation and Misclassification Analysis (Section 6.1, Table 11)

**Goal:** Provide concrete examples of explanations for correct and incorrect predictions.

**Cases analyzed:**
- Ascending stairs: correctly classified — attribution + spline analysis
- Sitting vs. standing: correctly separated — mid-layer neuron ϕ_j^{(2)}(u) behavior
- Sitting → standing misclassified — error analysis via attributions

**Dependencies:** Experiments 1–2, 5

---

## Experiment 12 — Interpretability Analysis: Intrinsic vs. Post-Hoc (Section 6.2)

**Goal:** Clarify distinction between intrinsic interpretability (KAN univariate functions) and post-hoc explanations (gradients, SHAP).

**Approach:** Analyze layer-wise spline functions at early, mid, and late layers; map neurons to semantic roles (motion detector, posture unit, stair detector).

**Dependencies:** Experiments 1–2, 5

---

## Experiment 13 — Practical Impact of KAN-Based Explanations (Section 6.3)

**Goal:** Demonstrate utility for model debugging, error analysis, and user trust.

**Workflows illustrated:**
- Neuron-level debugging via spline shape inspection
- Class-wise error analysis via mean activations
- User-facing explanation summaries from attribution maps

**Dependencies:** Experiments 1–2, 5, 11–12

---

## Cross-Cutting Dependencies

```
Experiment 1 (Seen-class HAR baselines)
    ├── Experiments 2–13 (all depend on KAN architecture existing)
    └── All experiments require:
        - Dataset download and preprocessing pipeline (Section 3.2)
        - Subject-disjoint splits
        - Normalization statistics from training set only

Experiment 2 (ZSL module) depends on Experiment 1
Experiment 3 depends on Experiment 2 (trained model)
Experiment 4 depends on Experiments 1–2
Experiment 5 depends on Experiments 1–2 (trained TinyKAN-HAR)
Experiment 6 depends on Experiments 1–2
Experiments 7a–7i depend on Experiments 1–2
Experiment 8 depends on Experiments 1–2 (ZSL module)
Experiment 9 depends on Experiments 1–2, 8
Experiment 10 depends on Experiments 1–2 (int8 model export)
Experiments 11–13 depend on Experiments 1–2, 5
```

---

## Dataset Acquisition

| Dataset | URL | Access Date |
|---------|-----|-------------|
| UCI HAR | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones | 16 Oct 2025 |
| WISDM | https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset | 16 Oct 2025 |
| PAMAP2 | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | 16 Oct 2025 |