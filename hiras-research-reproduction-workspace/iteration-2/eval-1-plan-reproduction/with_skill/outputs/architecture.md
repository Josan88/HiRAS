# Phase 2: Architecture Design — TinyKAN-HAR Reproduction

## Overview

The TinyKAN-HAR system consists of five major subsystems:
1. **Data Pipeline** — Dataset downloading, preprocessing, windowing, normalization
2. **KAN Backbone** — KAN layers with learnable spline nonlinearities
3. **ZSL Module** — Semantic embedding + compatibility function + calibration
4. **Explainability Layer** — Gradient attributions, SHAP, spline visualization
5. **TinyML Deployment** — Quantization, LUT generation, TFLM conversion, MCU profiling

---

## Directory Structure

```
tinykan-har/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── download.py              # Dataset download scripts
│   ├── uci_har/
│   ├── wisdm/
│   └── pamap2/
├── preprocessing/
│   ├── __init__.py
│   ├── temporal_align.py         # Resampling to 50 Hz
│   ├── gravity_separation.py     # FIR low-pass filter for accelerometer
│   ├── segmentation.py           # Windowing with stride
│   ├── normalization.py          # Per-channel z-score
│   ├── missing_values.py         # PAMAP2 gap interpolation
│   └── dataset.py                # Unified dataset class
├── models/
│   ├── __init__.py
│   ├── kan_layer.py              # Single KAN layer (W + b, spline φ)
│   ├── kan.py                    # Full KAN (L layers)
│   ├── classification_head.py    # Linear classification V + c
│   ├── zsl_module.py              # Semantic projection W_sem, compatibility
│   ├── tinykan_har.py            # Full model (KAN + cls head + ZSL)
│   └── baselines/
│       ├── __init__.py
│       ├── knn.py
│       ├── svm.py
│       ├── random_forest.py
│       ├── cnn1d.py
│       ├── lstm.py
│       ├── cnn_lstm.py
│       └── transformer.py
├── training/
│   ├── __init__.py
│   ├── train_kan.py              # Main training loop
│   ├── losses.py                 # L_CE, L_align, L_sem-CE, L_smooth
│   ├── scheduler.py              # Cosine annealing / step schedule
│   └── multi_seed.py             # Multi-seed training wrapper
├── evaluation/
│   ├── __init__.py
│   ├── har_metrics.py            # Accuracy, Precision, Recall, F1
│   ├── zsl_metrics.py            # Acc_ZSL, Acc_seen, Acc_unseen, H
│   ├── statistical_tests.py       # Paired t-test
│   └── ablations.py              # Ablation study runner
├── explainability/
│   ├── __init__.py
│   ├── gradient_attribution.py   # Gradient × input
│   ├── sensor_aggregation.py     # Per-sensor relevance
│   ├── temporal_aggregation.py    # Per-time relevance
│   ├── shap_approx.py             # SHAP-style global importance
│   └── spline_viz.py              # Extract and plot univariate splines
├── deployment/
│   ├── __init__.py
│   ├── quantize.py                # 8-bit symmetric quantization
│   ├── lut_generator.py          # LUT generation for spline evaluation
│   ├── tflm_converter.py          # Export to TensorFlow Lite Micro
│   └── mcu_profiler.py            # MCU latency/energy measurement
├── visualization/
│   ├── __init__.py
│   ├── attribution_heatmaps.py    # Figure 3 top row
│   ├── sensor_bars.py             # Figure 3 middle row
│   ├── temporal_curves.py         # Figure 3 bottom row
│   └── spline_plots.py            # Figure 4 spline functions
├── experiments/
│   ├── exp1_har_performance.py    # Table 3
│   ├── exp2_zsl.py                # Table 4
│   ├── exp3_gamma_robustness.py   # Table 5, Figure 2
│   ├── exp4_statistical.py        # Table 6
│   ├── exp5_ablations.py          # Tables 7, 8, 9
│   ├── exp6_tinyml.py             # Table 10
│   ├── exp7_viz.py                # Figures 3, 4
│   └── exp8_depth_width.py        # Section 5.6
├── scripts/
│   ├── download_data.sh
│   ├── train_all.sh
│   ├── run_experiments.sh
│   └── profile_mcu.sh
└── outputs/
    ├── tables/                    # CSV/LaTeX tables
    └── figures/                   # Generated plots
```

---

## Module Responsibilities

### Data Pipeline

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `preprocessing/temporal_align.py` | Resample channels to 50 Hz via linear interpolation or low-pass decimation | `resample_to_50hz(signal, orig_fs)` |
| `preprocessing/gravity_separation.py` | FIR low-pass filter to separate body/gravity acceleration | `separate_gravity(acc, fs, cutoff=0.3)` |
| `preprocessing/segmentation.py` | Sliding window segmentation with stride | `segment_windows(signal, T, S)` |
| `preprocessing/normalization.py` | Per-channel z-score normalization | `normalize_windows(windows, mu, sigma)` |
| `preprocessing/missing_values.py` | Linear interpolation of gaps, discard long gaps | `interpolate_gaps(signal, mask, max_gap_len)` |
| `preprocessing/dataset.py` | Unified HARDataset torch Dataset | `HARDataset(windows, labels, subjects)` |

### KAN Backbone

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `models/kan_layer.py` | Single KAN layer: linear mixing + B-spline univariate functions | `KANLayer(d_in, d_out, k_splines)` |
| `models/kan.py` | L-layer KAN with pyramidal width schedule | `KAN(input_dim, layer_dims, k_splines)` |
| `models/tinykan_har.py` | Full TinyKAN-HAR: KAN + cls head + ZSL module | `TinyKANHAR(config)` |

**KANLayer internal:**
- `W`: Linear mixing matrix (d_out × d_in)
- `b`: Bias vector (d_out)
- `theta`: Spline coefficients (d_out × K) per neuron
- `basis_fn`: B-spline basis function evaluator
- `forward(x)`: u = Wx + b; output = sum_k theta[:,k] * B_k(u)

### ZSL Module

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `models/zsl_module.py` | Semantic projection, compatibility scoring, calibration | `ZSLModule(latent_dim, semantic_dim, n_seen, gamma)` |

**ZSLModule internal:**
- `W_sem`: Learned projection (semantic_dim × latent_dim)
- `S`: Semantic embedding matrix (semantic_dim × n_classes)
- `compatibility(z, s_y)`: Cosine similarity g_phi(z, s_y)
- `calibrate_scores(g_seen, g_unseen)`: Subtract gamma from seen scores
- `forward_zsl(z)`: Returns predicted class among seen+unseen

### Training

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `training/losses.py` | L_CE, L_align, L_sem-CE, L_smooth, L_total | `cross_entropy_loss()`, `alignment_loss()`, `sem_ce_loss()`, `smoothness_loss()` |
| `training/train_kan.py` | Full training loop with early stopping | `train_epoch()`, `evaluate()`, `fit()` |
| `training/multi_seed.py` | Train with N random seeds, collect statistics | `train_multiple_seeds(model, seeds, ...)` |

### Evaluation

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `evaluation/har_metrics.py` | Accuracy, precision, recall, F1 (micro/macro) | `compute_har_metrics(y_true, y_pred)` |
| `evaluation/zsl_metrics.py` | Pure ZSL and gZSL metrics | `compute_zsl_metrics(model, seen_loader, unseen_loader)` |
| `evaluation/statistical_tests.py` | Paired t-test between model runs | `paired_ttest(results_a, results_b)` |

### Explainability

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `explainability/gradient_attribution.py` | Gradient × input attribution matrix | `compute_attribution(model, window, target_class)` |
| `explainability/sensor_aggregation.py` | Sum attributions over time per channel | `aggregate_sensor_attributions(attr_matrix)` |
| `explainability/temporal_aggregation.py` | Sum attributions over sensors per time step | `aggregate_temporal_attributions(attr_matrix)` |
| `explainability/shap_approx.py` | SHAP-style global feature importance | `compute_shap_importance(model, dataset, n_samples)` |
| `explainability/spline_viz.py` | Extract univariate splines and derivatives | `extract_spline_functions(model)` |

### Deployment

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `deployment/quantize.py` | Symmetric 8-bit quantization of weights/activations | `quantize_model(model, calib_data)` |
| `deployment/lut_generator.py` | Generate lookup tables for spline evaluation | `generate_spline_luts(model, grid_size)` |
| `deployment/tflm_converter.py` | Export to TensorFlow Lite Micro format | `convert_to_tflm(quantized_model)` |
| `deployment/mcu_profiler.py` | Measure latency/energy on target MCU | `profile_on_mcu(model_binary, mcu_config)` |

---

## Entry Points and Execution Flow

### Training Flow
```
python -m experiments.exp1_har_performance
    └── loads config.yaml
    └── for each dataset (UCI HAR, WISDM, PAMAP2):
        └── preprocess data → HARDataset
        └── for each model (kNN, SVM, RF, 1D-CNN, LSTM, CNN-LSTM, Transformer, TinyKAN-HAR):
            └── train on seen classes
            └── evaluate on test split
            └── save metrics to outputs/tables/
```

### ZSL Flow
```
python -m experiments.exp2_zsl
    └── train TinyKAN-HAR with ZSL losses
    └── generate semantic embeddings (ATTR / TEXT / HYBRID)
    └── evaluate pure ZSL on unseen classes
    └── evaluate gZSL with calibration
    └── save Table 4 metrics
```

### Ablation Flow
```
python -m experiments.exp5_ablations
    └── for each of 20 Table 7 variants:
        └── retrain with specific configuration
        └── profile TinyML metrics
        └── collect Acc, F1, Acc_ZSL, H, model size, latency
    └── for Table 8: retrain with ATTR/TEXT/HYBRID embeddings
    └── for Table 9: retrain with RANDOM/SHUFFLED semantics
```

---

## External Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| PyTorch | Deep learning framework | ≥2.0 |
| NumPy | Array operations | ≥1.24 |
| SciPy | Signal processing (filtering), statistics | ≥1.10 |
| scikit-learn | kNN, SVM, RF baselines | ≥1.3 |
| TensorFlow Lite Micro | TinyML conversion | (build from source or CMSIS-NN) |
| matplotlib | Visualization | ≥3.7 |
| pandas | Result tabulation | ≥2.0 |

**Note on TinyML:** Actual MCU profiling (Table 10) requires ARM Cortex-M4F hardware or QEMU emulation. Without hardware, latency/energy estimates can be computed analytically from op count × MCU clock estimates.

---

## Traceability: Experiments → Modules

| Experiment | Required Modules |
|------------|-----------------|
| 1 (HAR performance) | `dataset.py`, `kan.py`, `tinykan_har.py`, `baselines/*`, `har_metrics.py`, `train_kan.py` |
| 2 (ZSL) | `zsl_module.py`, `semantic_embeddings.py` (ATTR/TEXT/HYBRID), `zsl_metrics.py` |
| 3 (γ robustness) | `zsl_module.py` (gamma sweep), `zsl_metrics.py`, `visualization/gamma_sweep.py` |
| 4 (Statistical sig.) | `multi_seed.py`, `statistical_tests.py` |
| 5A (Table 7) | `kan_layer.py` (depth/width variants), `quantize.py`, `lut_generator.py`, `mcu_profiler.py` |
| 5B (Table 8) | Semantic embedding generation, retraining |
| 5C (Table 9) | Random/shuffled embedding generation, retraining |
| 6 (TinyML) | `quantize.py`, `lut_generator.py`, `tflm_converter.py`, `mcu_profiler.py` |
| 7 (Visualization) | `gradient_attribution.py`, `sensor_aggregation.py`, `temporal_aggregation.py`, `spline_viz.py` |
| 8 (Depth/width) | `kan_layer.py` (L, d_L variants) |

Every experiment from `plan.md` is traceable to at least one module above. No orphaned experiments.
