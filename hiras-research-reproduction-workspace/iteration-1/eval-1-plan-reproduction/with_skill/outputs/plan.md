# Phase 1: Overall Planning - TinyKAN-HAR Reproduction

## Paper Overview
- **Title**: Explainable Kolmogorov-Arnold Networks for Zero-Shot Human Activity Recognition on TinyML Edge Devices (TinyKAN-HAR)
- **Core Question**: Can a KAN-based HAR model achieve high accuracy, zero-shot recognition of unseen activities, and TinyML-deployable interpretability simultaneously?
- **Datasets**: UCI HAR (30 subjects, 6 activities, 50Hz), WISDM (51 subjects, 18 activities, 20Hz), PAMAP2 (9 subjects, 18 activities, 100Hz→50Hz)
- **Key Claim**: >97% macro-F1 on seen classes, >96% accuracy on unseen, 145kB flash, 26kB RAM, 4.1ms latency on Cortex-M4F

---

## Experiment Inventory

### Experiment 1: HAR Performance on Seen Classes (Section 5.1, Table 3)
**Goal**: Evaluate TinyKAN-HAR vs. baselines (kNN, SVM, RF, 1D-CNN, LSTM, CNN-LSTM, Transformer) on supervised classification of seen activities.

**Datasets**: UCI HAR, WISDM, PAMAP2  
**Protocol**: Subject-disjoint splits; train on seen classes only  
**Metrics**: Overall accuracy, macro-F1

**Details**:
- UCI HAR: 21 train / 4 val / 9 test; seen={WALKING, SITTING, STANDING, LAYING}, unseen={WALKING_UPSTAIRS, WALKING_DOWNSTAIRS}
- WISDM: 35 train / 8 val / 8 test; seen=14 activities, unseen={jogging, walking_upstairs, walking_downstairs, jumping}
- PAMAP2: 6 train / 1 val / 2 test; seen=14 activities, unseen={running, rope_jumping, vacuum_cleaning, ironing}

**Dependencies**: None (standalone experiment)

---

### Experiment 2: Zero-Shot and Generalized Zero-Shot Performance (Section 5.2, Table 4)
**Goal**: Evaluate pure ZSL and gZSL capability of TinyKAN-HAR with semantic embedding module.

**Datasets**: UCI HAR, PAMAP2  
**Protocol**: Pure ZSL (unseen classes only) and gZSL (seen + unseen with calibration factor γ=0.5)  
**Metrics**: Acc_ZSL, Acc_seen, Acc_unseen, harmonic mean H

**Details**:
- Semantic embeddings: hybrid (attributes + text embeddings)
- Calibration factor γ=0.5 for gZSL
- Baselines: CNN+ZSL head, LSTM+ZSL, Transformer+ZSL head

**Dependencies**: Requires trained TinyKAN-HAR with ZSL module from Experiment 1

---

### Experiment 3: Robustness of Calibration Factor γ (Section 5.3, Table 5, Figure 2)
**Goal**: Verify that gZSL performance is robust to γ choice, not over-tuned.

**Dataset**: UCI HAR  
**Protocol**: Sweep γ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}; evaluate on val and test  
**Metrics**: Acc_ZSL, Acc_seen, Acc_unseen, H

**Dependencies**: Trained TinyKAN-HAR from Experiment 1 (same model, different inference gamma)

---

### Experiment 4: Statistical Significance Across Random Seeds (Section 5.4, Table 6)
**Goal**: Confirm TinyKAN-HAR gains are statistically significant, not due to lucky initialization.

**Datasets**: UCI HAR, PAMAP2  
**Protocol**: 5 random seeds per model; paired two-sided t-test on H  
**Metrics**: Mean ± std of accuracy and H; p-value

**Dependencies**: Trained models from Experiment 1 (5 runs each for KAN-HAR and Transformer+ZSL)

---

### Experiment 5: Ablation Studies (Section 5.7, Tables 7, 8, 9)
**Goal**: Quantify contribution of each architectural/training choice.

**Dataset**: UCI HAR (primary), PAMAP2 (for semantic embedding ablation)

**Sub-experiments**:
- 5a: ZSL losses vs. w/o ZSL losses (Table 7 rows 2-3)
- 5b: Calibration vs. w/o calibrated scores (Table 7 rows 2, 4)
- 5c: Semantic projection layer vs. w/o (Table 7 rows 2, 5)
- 5d: Explainability regularizer vs. w/o (Table 7 rows 2, 6)
- 5e: KAN depth (L=1,3,4) and latent dim (d_L=64,128,256) (Table 7 rows 7-10)
- 5f: Spline resolution (coarse/fine knots) (Table 7 rows 11-12)
- 5g: LUT-based spline evaluation vs. direct computation (Table 7 rows 13-14)
- 5h: Structured pruning (50%, 70%) (Table 7 rows 15-16)
- 5i: Window length (short/long T) (Table 7 rows 17-18)
- 5j: Dropout rate (p=0.1, 0.5) (Table 7 rows 19-20)
- 5k: Semantic representation (ATTR/TEXT/HYBRID) on PAMAP2 (Table 8)
- 5l: Semantic structure ablation (HYBRID/RANDOM/SHUFFLED) (Table 9)

**Dependencies**: Full TinyKAN-HAR architecture from Experiment 1

---

### Experiment 6: TinyML Deployment Metrics (Section 5.8, Table 10)
**Goal**: Quantify memory footprint, latency, and energy on Cortex-M4F MCU.

**Protocol**: Measure int8 quantized models on target hardware  
**Metrics**: Flash (kB), RAM (kB), latency (ms), energy (μJ)

**Details**:
- Target: Cortex-M4F, 256kB flash, 64kB SRAM, 80MHz
- TinyKAN-HAR Tiny: 145kB flash, 26kB RAM, 4.1ms, 320μJ
- Compare against 1D-CNN Tiny, CNN-LSTM Tiny, Transformer Tiny

**Dependencies**: Quantization and model conversion pipeline (Section 3.6)

---

## Implementation Roadmap

### Stage 1: Data Pipeline & Preprocessing
1. Download and parse UCI HAR, WISDM, PAMAP2 datasets
2. Implement temporal alignment and resampling to 50Hz
3. Implement gravity separation (low-pass filter, 0.3-0.5Hz cutoff)
4. Implement windowing (T=128/200/250 samples, 50% overlap)
5. Implement per-channel z-score normalization (train-only stats)
6. Implement missing value interpolation (PAMAP2)
7. Implement seen/unseen class splits

### Stage 2: KAN Feature Extractor
1. Implement KAN layer (linear mixing + univariate B-spline functions)
2. Implement spline coefficient initialization (identity approximation)
3. Implement L-layer KAN stack with configurable widths
4. Implement classification head (linear + softmax)
5. Implement cross-entropy loss
6. Implement regularization: weight decay, spline smoothness, dropout

### Stage 3: Zero-Shot Learning Module
1. Implement semantic embedding generation (attributes + text encoder)
2. Implement semantic projection layer W_sem
3. Implement alignment loss (L2 distance)
4. Implement semantic softmax loss
5. Implement combined ZSL loss with λ parameters
6. Implement pure ZSL and gZSL inference with calibration

### Stage 4: Training Infrastructure
1. Implement Adam optimizer with learning rate scheduling
2. Implement early stopping on validation macro-F1
3. Implement multi-seed training for statistical significance
4. Implement experiment 1 training loop for all baselines

### Stage 5: Baselines
1. Implement kNN baseline (feature extraction + sklearn)
2. Implement SVM baseline (feature extraction + sklearn)
3. Implement Random Forest baseline (feature extraction + sklearn)
4. Implement 1D-CNN baseline (PyTorch)
5. Implement LSTM baseline (PyTorch)
6. Implement CNN-LSTM baseline (PyTorch)
7. Implement Transformer baseline (PyTorch)
8. Implement ZSL head variants for CNN/LSTM/Transformer

### Stage 6: Explainability
1. Implement gradient-based attributions (gradient × input)
2. Implement sensor-level aggregation
3. Implement temporal-level aggregation
4. Implement SHAP-style global importance (sampling-based)
5. Implement univariate spline function visualization

### Stage 7: TinyML Optimization
1. Implement uniform weight quantization (int8)
2. Implement activation quantization
3. Implement LUT generation for spline functions
4. Implement structured pruning
5. Implement model export to TFLM format
6. Measure deployment metrics on target MCU (or simulator)

### Stage 8: Experiments Execution
1. Run Experiment 1 (seen-class HAR)
2. Run Experiment 2 (ZSL/gZSL)
3. Run Experiment 3 (γ sensitivity)
4. Run Experiment 4 (statistical significance)
5. Run Experiment 5 (ablations)
6. Run Experiment 6 (TinyML metrics)

---

## Expected Outcomes (per paper)
- **Experiment 1**: TinyKAN-HAR ≥ 98.3% accuracy, 98.0% macro-F1 on UCI HAR
- **Experiment 2**: Acc_ZSL ≥ 96.4%, H ≥ 96.7% on UCI HAR
- **Experiment 3**: H ≥ 96% for γ ∈ [0.25, 1.0]
- **Experiment 4**: p-value < 0.01 for KAN-HAR vs Transformer+ZSL on H
- **Experiment 5**: All ablations match paper Table 7/8/9 within tolerance
- **Experiment 6**: 145kB flash, 26kB RAM, 4.1ms latency, 320μJ energy

---

## Ambiguities & Missing Information
1. **Text embedding model**: Paper mentions "pretrained language model" but does not specify which one (e.g., Word2Vec, GloVe, BERT). Will use a common choice and note as UNKNOWN.
2. **Attribute definitions**: Paper mentions "manually defined binary properties" but does not list them. Will infer from activity names.
3. **Specific optimizer learning rate schedule**: Paper gives range but not exact schedule (e.g., "cosine annealing" mentioned). Will use paper's suggested range and note as TODO.
4. **Batch size**: Paper gives range {32, 64, 128} but does not specify per-dataset. Will tune on validation.
5. **Number of B-spline knots (K)**: Not explicitly stated for coarse/fine variants. Will infer from model sizes in Table 7.
6. **WISDM ZSL results**: Table 4 does not include WISDM ZSL results. Only UCI HAR and PAMAP2 are shown.
7. **Exact random seeds**: Not specified. Will use standard seeds {42, 43, 44, 45, 46}.