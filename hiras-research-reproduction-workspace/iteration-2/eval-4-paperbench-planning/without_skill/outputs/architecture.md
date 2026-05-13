# Architecture: TinyKAN-HAR — Explainable KAN for Zero-Shot HAR on TinyML

## Top-Level Directory Structure

```
tinykan_har/
├── data/
│   ├── __init__.py
│   ├── datasets.py           # Dataset loaders for UCI HAR, WISDM, PAMAP2
│   ├── preprocessing.py      # Sections 3.2.1–3.2.7: resampling, segmentation,
│   │                         #   gravity separation, normalization, seen/unseen splits
│   └── transforms.py         # Augmentation transforms (optional)
├── models/
│   ├── __init__.py
│   ├── kan_layer.py          # Single KAN layer: linear mixing + univariate splines
│   ├── kan.py                # Full KAN feature extractor (Sections 3.3.1–3.3.6)
│   ├── classification_head.py  # Linear classification head (Section 3.3.4)
│   ├── zsl_module.py         # Zero-shot learning module (Sections 3.4.1–3.4.7)
│   ├── explainability.py     # Multi-level explainability (Section 3.5)
│   ├── tinykan_har.py        # Full TinyKAN-HAR model integrating all components
│   └── baselines/
│       ├── __init__.py
│       ├── classical.py      # kNN, SVM, Random Forest (Section 4.1.1)
│       ├── cnn.py            # 1D-CNN baseline (Section 4.1.2)
│       ├── lstm.py           # LSTM and CNN-LSTM baselines (Section 4.1.2)
│       └── transformer.py    # Transformer encoder baseline (Section 4.1.2)
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Main training loop with Adam optimizer, early stopping
│   ├── loss.py              # Loss functions: CE, ZSL alignment, semantic softmax, regularization
│   ├── optimizer.py         # Adam optimizer with learning rate scheduling
│   └── callbacks.py         # Early stopping, model checkpointing, metric logging
├── quantization/
│   ├── __init__.py
│   ├── quantize.py          # Uniform int8/FP32 quantization (Section 3.6.1)
│   ├── quantize_kan.py      # KAN-specific quantization (weight + activation)
│   └── qat.py              # Quantization-aware training (optional fine-tuning)
├── tinyml/
│   ├── __init__.py
│   ├── export.py           # Export to TFLite/ONNX for MCU deployment (Section 3.6.2)
│   ├── lut.py             # Build lookup tables for spline functions (Section 3.6.3)
│   ├── benchmark.py       # On-device benchmarking on Cortex-M4F (Section 5.8)
│   └── cortex_m4/         # C header files / firmware template for MCU
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py         # Classification metrics, ZSL metrics, TinyML metrics
│   ├── explainability_eval.py  # Attribution map generation, spline visualization
│   └── ablations.py      # Ablation study runner (Experiment 7)
├── configs/
│   └── config.yaml        # Hyperparameters and experiment configuration
├── scripts/
│   ├── train_seen.py      # Train TinyKAN-HAR on seen classes (Experiments 1, 6)
│   ├── train_zsl.py       # Train with ZSL module (Experiments 2, 3, 4)
│   ├── eval_seen.py       # Evaluate seen-class HAR performance (Experiment 1)
│   ├── eval_zsl.py        # Evaluate ZSL and gZSL performance (Experiment 2)
│   ├── eval_calibration.py # Sweep calibration factor γ (Experiment 3)
│   ├── eval_multi_seed.py  # Statistical significance across seeds (Experiment 4)
│   ├── visualize_explanations.py  # Generate explanation visualizations (Experiment 5)
│   ├── run_ablations.py   # Run all ablation configurations (Experiments 7a–7i)
│   ├── eval_semantic_ablation.py  # Ablation on semantic representations (Experiment 8)
│   ├── eval_semantic_structure.py  # RANDOM/SHUFFLED semantic ablation (Experiment 9)
│   ├── deploy_tinyml.py   # Export, build, flash, and benchmark on MCU (Experiment 10)
│   └── analyze_interpretations.py  # Qualitative interpretation analysis (Experiments 11–13)
├── requirements.txt
├── README.md
└── main.py                # Entry point: orchestrates all experiments
```

---

## Module Responsibilities

### data/datasets.py
- Download (if not present) and load UCI HAR, WISDM, PAMAP2
- Expose a unified `HARDataset` class with `train/val/test` splits
- Handle subject-disjoint splits as specified in Section 3.1
- Provide seen/unseen class partition per dataset

### data/preprocessing.py
- Temporal alignment and resampling to 50 Hz target (Eq. 1)
- Segmentation into fixed-length overlapping windows (Eq. 2, T per dataset)
- Gravity separation via FIR low-pass filter (Eqs. 4–5)
- Per-channel z-score normalization using training statistics only (Eq. 6)
- Missing value interpolation and artefact handling (Eq. 8)
- Construction of seen/unseen window subsets (Eq. 9)
- Final vectorized input representation (Eq. 10)

### models/kan_layer.py
- `KANLayer`: linear mixing W^{(l)} + b^{(l)}, then bank of univariate spline functions ϕ_j^{(l)}(u)
- Spline evaluation via B-spline basis (Eq. 13): K fixed basis functions, θ_j,k learnable coefficients
- Methods: forward, compute_spline_smoothness_reg (Eq. 21)
- Support dropout (Eq. 23)

### models/kan.py
- `KANFeatureExtractor`: sequence of L KAN layers
- Pyramidal width schedule: d_0 > d_1 > ... > d_L
- Output: latent vector z = x^{(L)} ∈ ℝ^{d_L} (Eq. 15)
- Regularization: weight decay on W^{(l)} and V, smoothness penalty on spline coefficients

### models/zsl_module.py
- `ZSLModule`: semantic embedding lookup (Eq. 26), learned projection W_sem (Eq. 28)
- Alignment loss L_align (Eq. 30)
- Semantic softmax loss L_sem-CE (Eq. 35)
- Combined ZSL loss L_ZSL (Eq. 36)
- Cosine compatibility function g_φ (Eq. 31)
- Pure ZSL inference (Eq. 38) and gZSL with score calibration γ (Eq. 40)

### models/explainability.py
- `ExplainabilityModule`: local gradient-based attributions (gradient × input)
- Sensor-level aggregation (per-channel relevance R_i[d])
- Temporal aggregation (per time-step relevance R_i[t])
- SHAP-style global feature importance
- Global spline function inspection and derivative analysis

### models/tinykan_har.py
- `TinyKANHAR`: integrates KANFeatureExtractor + ClassificationHead + ZSLModule + ExplainabilityModule
- Full forward pass for seen-class training, ZSL training, and inference
- Combined loss: L_task = L_CE + λ_ZSL * L_ZSL + R_KAN (Eqs. 18–25)

### quantization/quantize.py
- Uniform symmetric quantization to b ∈ {8, 16} bits (Eq. 42)
- Post-training quantization with optional fine-tuning
- Per-layer/tensor scale factors Δ
- Activation quantization with layer-wise scaling (Eq. 580)

### quantization/lut.py
- Build precomputed lookup tables for each univariate spline function
- Uniform grid sampling over pre-activation range
- Quantize LUT entries to int8
- Integer-only inference via table lookup + linear interpolation

### tinyml/export.py
- Export trained model to TFLite format
- Embed quantized weights and LUT arrays as static C arrays
- Generate self-contained firmware image for Cortex-M4F

### evaluation/metrics.py
- Seen-class: accuracy, precision, recall, F1 (macro/micro)
- ZSL: Acc_ZSL (pure), Acc_seen, Acc_unseen, harmonic mean H (gZSL)
- TinyML: model size (kB), peak RAM (kB), latency (ms), energy (μJ)
- Statistical significance: paired t-test on H across seeds

---

## Entry Point and Execution Flow

```
main.py
├── argparse: experiment index {1..13} or "all"
├── load config from configs/config.yaml
├── setup random seed
├── instantiate data preprocessing pipeline
├── run selected experiment(s)
│   ├── Experiment 1: train_all_baselines() → compare against paper Table 3
│   ├── Experiment 2: train_and_eval_zsl() → compare against paper Table 4
│   ├── Experiment 3: eval_calibration_sensitivity() → compare against paper Table 5
│   ├── Experiment 4: eval_multi_seed_significance() → compare against paper Table 6
│   ├── Experiment 5: visualize_explanations() → reproduce Figures 3–4
│   ├── Experiment 6: eval_kan_depth_width() → compare against Section 5.6
│   ├── Experiment 7: run_extended_ablations() → reproduce Table 7
│   ├── Experiment 8: eval_semantic_representations() → reproduce Table 8
│   ├── Experiment 9: eval_semantic_structure_ablation() → reproduce Table 9
│   ├── Experiment 10: deploy_and_benchmark_tinyml() → reproduce Table 10
│   ├── Experiment 11: analyze_misclassifications() → reproduce Table 11
│   ├── Experiment 12: intrinsic_vs_posthoc() → confirm Section 6.2 findings
│   └── Experiment 13: practical_explanation_utility() → confirm Section 6.3 findings
└── save results to results/<experiment_name>/
```

---

## External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Deep learning framework |
| torchvision | ≥ 0.15 | Dataset utilities |
| NumPy | ≥ 1.24 | Numerical computing |
| pandas | ≥ 2.0 | DataFrame utilities for metrics |
| scikit-learn | ≥ 1.3 | kNN, SVM, Random Forest baselines |
| scipy | ≥ 1.11 | Scientific computing (interpolation for resampling) |
| matplotlib | ≥ 3.7 | Visualization of attributions and splines |
| seaborn | ≥ 0.12 | Styled statistical plots |
| PyTorch::torch.quantization | built-in | Quantization utilities |
| tqdm | ≥ 4.66 | Progress bars |

---

## Key Interfaces

### KAN Layer → KAN Feature Extractor
- Input: x^{(l-1)} ∈ ℝ^{d_{l-1}}
- Output: x^{(l)} ∈ ℝ^{d_l}
- Each coordinate j independently transformed by univariate spline ϕ_j^{(l)}(u_j^{(l)})

### KAN Feature Extractor → ZSL Module
- Input: latent vector z ∈ ℝ^{d_L}
- ZSLModule projects z → h = W_sem * z ∈ ℝ^m (Eq. 28)
- h compared via cosine similarity with semantic embeddings s_y

### KAN Feature Extractor → Classification Head
- Input: latent vector z ∈ ℝ^{d_L}
- Linear classification: logits = V * z + c (Eq. 16)
- Softmax probabilities via Eq. 17

### Full Model → Explainability
- Gradient computed w.r.t. class score or semantic score
- Attributions reshaped to (T, D) heatmap over time × sensors
- Spline functions sampled and plotted per layer/neuron