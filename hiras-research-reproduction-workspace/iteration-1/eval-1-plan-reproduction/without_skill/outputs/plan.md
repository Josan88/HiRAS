# Plan: TinyKAN-HAR Reproduction

## 0. Paper Summary

**Paper:** "Explainable Kolmogorov-Arnold Networks for Zero-Shot Human Activity Recognition on TinyML Edge Devices" (TinyKAN-HAR)
**Core claim:** A KAN-based HAR model with zero-shot learning and explainability, deployable on TinyML microcontrollers (Cortex-M4F, 256kB flash, 64kB SRAM).

**Key contributions:**
1. KAN-based feature extractor replacing fixed activations with learnable 1D splines
2. Semantic-embedding-based ZSL module using cosine similarity and score calibration
3. Multi-level explainability: gradient attributions, SHAP, spline visualization
4. TinyML optimization: 8-bit quantization, LUT-based spline evaluation, structured pruning

**Datasets:** UCI HAR, WISDM, PAMAP2

---

## 1. Experiments to Reproduce

### Experiment 1: HAR Performance on Seen Classes (Section 5.1, Table 3)
- **Goal:** Compare TinyKAN-HAR against kNN, SVM, Random Forest, 1D-CNN, LSTM, CNN-LSTM, Transformer on seen classes
- **Datasets:** UCI HAR (30 subjects, 6 activities, 50Hz, 6 channels, window=128), WISDM (51 subjects, 18 activities, 20Hz, 12 channels, window=200), PAMAP2 (9 subjects, 18 activities, 100→50Hz, 28 channels, window=250)
- **Metrics:** Overall accuracy %, macro-F1 %
- **Split:** Subject-disjoint (train/val/test by subject), ZSL splits (seen vs unseen classes)
- **Expected outcomes:** TinyKAN-HAR should achieve ≥98.3% Acc, ≥98.0% F1 (UCI), ≥97.9% Acc, ≥97.7% F1 (WISDM), ≥97.3% Acc, ≥97.1% F1 (PAMAP2)

### Experiment 2: Zero-Shot and Generalized Zero-Shot Performance (Section 5.2, Table 4)
- **Goal:** Evaluate pure ZSL (unseen classes only) and gZSL (seen+unseen) with calibration factor γ
- **Datasets:** UCI HAR (seen: WALKING, SITTING, STANDING, LAYING; unseen: WALKING_UPSTAIRS, WALKING_DOWNSTAIRS), PAMAP2 (seen: 14 activities; unseen: running, rope_jumping, vacuum_cleaning, ironing)
- **Baselines:** CNN+ZSL, LSTM+ZSL, Transformer+ZSL
- **Metrics:** Acc_ZSL (pure), Acc_seen, Acc_unseen, Harmonic mean H (gZSL)
- **Expected outcomes:** TinyKAN-HAR ≥96.4% Acc_ZSL, ≥96.7% H (UCI); ≥96.0% Acc_ZSL, ≥96.0% H (PAMAP2)

### Experiment 3: Robustness of Calibration Factor γ (Section 5.3, Table 5, Figure 2)
- **Goal:** Sweep γ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} to verify ZSL robustness
- **Dataset:** UCI HAR
- **Metrics:** Acc_ZSL, Acc_seen, Acc_unseen, H per γ
- **Expected outcome:** Acc_ZSL and H stay above 96% for γ ∈ [0.25, 1.0]; γ*=0.5 chosen on validation generalizes to test

### Experiment 4: Statistical Significance Across Random Seeds (Section 5.4, Table 6)
- **Goal:** 5-run multi-seed evaluation comparing TinyKAN-HAR vs Transformer+ZSL
- **Datasets:** UCI HAR, PAMAP2
- **Metrics:** mean±std of Acc and H; paired two-sided t-test p-value on H
- **Expected outcomes:** KAN-HAR: Acc≥98.3±0.1%, H≥96.4±0.3%; Transformer+ZSL: Acc≥97.7±0.2%, H≥93.3±0.4%; p<0.01

### Experiment 5: Ablation Studies (Section 5.7, Table 7)
- **Goal:** Quantify contribution of each component on UCI HAR
- **Variants:** Full TinyKAN-HAR (int8, FP32), w/o ZSL losses, w/o calibrated scores, w/o semantic projection layer, w/o explainability, Shallow KAN (L=1,dL=64), Deep KAN (L=4,dL=128), Narrow latent (dL=64), Wide latent (dL=256), Coarse spline, Fine spline, w/o LUTs, Quantization only, Quant+50% pruning, Quant+70% pruning, Short window, Long window, Low dropout (p=0.1), High dropout (p=0.5)
- **Metrics:** Acc, F1 macro, Acc_ZSL, H, Model [kB], RAM [kB], Latency [ms], Energy [µJ]
- **Expected outcomes:** Full int8 model: 98.3% Acc, 145kB flash, 26kB RAM, 4.1ms latency, 320µJ

### Experiment 5b: Semantic Embedding Ablation (Section 5.7.1, Table 8)
- **Variants:** ATTR (manual attributes), TEXT (text embeddings), HYBRID (concat)
- **Expected outcomes:** HYBRID best: ≥96.8% Acc_ZSL, ≥97.1% H (UCI); ≥96.3% Acc_ZSL, ≥96.6% H (PAMAP2)

### Experiment 5c: Semantic Structure Isolation (Section 5.7.2, Table 9)
- **Variants:** HYBRID (meaningful), RANDOM (Gaussian), SHUFFLED (permuted labels)
- **Expected outcomes:** RANDOM/SHUFFLED collapse to ~20-25% Acc_ZSL; seen Acc stays ~98% for all

### Experiment 6: TinyML Deployment Metrics (Section 5.8, Table 10)
- **Goal:** On-device benchmarks on Cortex-M4F @80MHz, 256kB flash, 64kB SRAM
- **Models:** TinyKAN-HAR, 1D-CNN, CNN-LSTM, Transformer (all int8)
- **Metrics:** Acc [%], Model [kB], Peak RAM [kB], Latency [ms], Energy [µJ]
- **Expected outcomes:** TinyKAN-HAR: 98.3% Acc, 145kB, 26kB RAM, 4.1ms, 320µJ

### Experiment 7: KAN Depth/Latent Dimension Analysis (Section 5.6, Table 7)
- **Goal:** Analyze L ∈ {1,2,3,4} and dL ∈ {64,128,256}
- **Expected:** L=3, dL=128 is optimal trade-off; L=4 marginal gain over L=3

### Experiment 8: Case Studies and Explanation Visualization (Section 5.5, Figures 3-4)
- **Goal:** Generate attribution heatmaps, sensor/temporal relevance plots, spline function plots for walking, sitting, ascending stairs
- **Outputs:** Figures showing attribution matrices, sensor-level relevance, temporal relevance curves, univariate spline functions per layer

---

## 2. Implementation Phases

### Phase A: Environment & Data Setup
1. Install PyTorch, numpy, scipy, scikit-learn, matplotlib, pandas
2. Download UCI HAR (30 subjects, 6 activities, 50Hz, phone Acc+Gyro)
3. Download WISDM (51 subjects, 18 activities, 20Hz, phone+watch Acc+Gyro)
4. Download PAMAP2 (9 subjects, 18 activities, 100Hz→50Hz, 3 IMUs+HR, 28 channels)
5. Implement preprocessing pipeline: temporal alignment, resampling to 50Hz, gravity separation, per-channel z-score normalization, window segmentation (50% overlap), majority-vote labels

### Phase B: KAN Implementation
1. Implement spline basis functions (B-splines, K knots per neuron)
2. Implement KAN layer: linear mixing W(l) + bias b(l) → univariate spline φ(l)_j(u)
3. Implement multi-layer KAN architecture (L=3, dL=128 default)
4. Implement classification head (linear V + bias c)
5. Implement regularization: weight decay, spline smoothness penalty, dropout

### Phase C: ZSL Module Implementation
1. Implement semantic embedding generation (attributes, text embeddings, hybrid)
2. Implement semantic projection layer W_sem
3. Implement cosine similarity compatibility function g_φ(z, s_y)
4. Implement alignment loss L_align and semantic cross-entropy L_sem-CE
5. Implement score calibration with γ parameter
6. Implement pure ZSL and gZSL inference

### Phase D: Training Pipeline
1. Implement data loaders with subject-based splits
2. Implement Adam optimizer with learning rate schedule (step or cosine annealing)
3. Implement early stopping on validation macro-F1
4. Implement multi-seed training (5 seeds for significance tests)
5. Implement hyperparameter configuration from config.yaml

### Phase E: Baselines Implementation
1. Implement classical baselines: kNN, SVM (RBF), Random Forest with hand-crafted features
2. Implement 1D-CNN baseline
3. Implement LSTM baseline
4. Implement CNN-LSTM baseline
5. Implement Transformer baseline (positional encoding, multi-head self-attention, FFN)
6. Implement ZSL head on deep baselines (CNN+ZSL, LSTM+ZSL, Transformer+ZSL)

### Phase F: Evaluation & Metrics
1. Implement classification metrics: accuracy, precision, recall, F1 (macro/micro)
2. Implement ZSL metrics: Acc_ZSL, Acc_seen, Acc_unseen, harmonic mean H
3. Implement statistical significance testing (paired t-test across seeds)
4. Implement ablation study runner

### Phase G: Explainability Module
1. Implement gradient-based attributions (grad × input)
2. Implement sensor-level aggregation
3. Implement temporal-level aggregation
4. Implement SHAP-style global importance (sampling-based)
5. Implement spline function visualization
6. Generate explanation figures

### Phase H: TinyML Deployment
1. Implement 8-bit quantization (uniform symmetric)
2. Implement activation quantization with per-layer scaling
3. Implement LUT generation for spline functions
4. Implement structured pruning (50%, 70%)
5. Estimate/model TinyML metrics (flash, RAM, latency, energy)

---

## 3. Dependencies Between Experiments

- Phase A must complete before all experiments
- Phase B (KAN core) must complete before Phase C (ZSL) and Phase D (training)
- Phase E (baselines) can run in parallel with Phase B/C after Phase A
- Phase F (evaluation) depends on Phase D and Phase E
- Phase G (explainability) depends on Phase B and Phase D
- Phase H (TinyML) depends on Phase B, C, D

---

## 4. Validation Gates

1. **Data gate:** All three datasets downloaded, preprocessed, and verified (subject splits, window counts, label distributions)
2. **KAN gate:** KAN layer produces correct shapes; spline evaluation works; gradient backpropagation verified
3. **ZSL gate:** Semantic embeddings load correctly; alignment loss decreases during training; calibration γ works
4. **Training gate:** Training converges; seen-class accuracy ≥96% on UCI HAR; early stopping triggers reasonably
5. **Ablation gate:** All 20+ ablation variants can be run from same codebase via config flags
6. **TinyML gate:** Quantized model produces same accuracy as FP32 (within 0.2%); LUT sizes match expected kB values

---

## 5. Known Ambiguities and TODOs

1. **WISDM ZSL results missing from Table 4** — paper only reports UCI HAR and PAMAP2 for ZSL
2. **Table 4 formatting** — "Acc ZSL" column appears twice; second column likely WISDM but values not clearly attributed
3. **Exact spline knot count** — not explicitly stated; infer from model size (145kB flash)
4. **Learning rate schedule specifics** — paper says "step schedule or cosine annealing" but no exact values
5. **Bayesian optimization for hyperparameters** — mentioned but not detailed
6. **Exact number of training epochs** — "100-200" but no definitive value
7. **Text embedding model** — "pretrained language model" but specific model not named
8. **SHAP sampling count** — "sampling-based estimator" but no detail on number of samples
9. **Figure 2 y-axis label** — appears truncated in paper; unclear what "97.5", "96.0" etc. represent exactly
10. **PAMAP2 ZSL rows in Table 4** — only 2 datasets shown (UCI, PAMAP2); WISDM ZSL row appears merged
11. **Code availability** — no GitHub link or supplementary code provided in paper