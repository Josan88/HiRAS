# Phase 2: Architecture Design - TinyKAN-HAR Reproduction

## Directory Layout

```
tinykan_har/
├── __init__.py
├── config.py                 # Configuration loader (yaml)
├── config.yaml               # Hyperparameters and settings
├── datasets/
│   ├── __init__.py
│   ├── base.py               # Abstract dataset class
│   ├── uci_har.py            # UCI HAR dataset
│   ├── wisdm.py             # WISDM dataset
│   ├── pamap2.py            # PAMAP2 dataset
│   └── preprocessor.py      # Shared preprocessing pipeline
├── models/
│   ├── __init__.py
│   ├── kan_layer.py         # KAN layer (linear mix + B-spline)
│   ├── kan.py               # KAN feature extractor (L layers)
│   ├── classifier.py        # Linear classification head
│   ├── zsl_module.py        # Zero-shot learning module
│   ├── explainer.py         # Explainability layer
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── knn.py           # k-Nearest Neighbors
│   │   ├── svm.py           # Support Vector Machine
│   │   ├── random_forest.py # Random Forest
│   │   ├── cnn1d.py         # 1D-CNN
│   │   ├── lstm.py          # LSTM
│   │   ├── cnn_lstm.py      # CNN-LSTM
│   │   └── transformer.py   # Transformer encoder
│   └── tinykan_har.py       # Full TinyKAN-HAR model
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Training loop
│   ├── optimizer.py         # Adam optimizer + scheduler
│   ├── losses.py            # CE + ZSL losses
│   └── early_stopping.py    # Early stopping logic
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # Accuracy, F1, ZSL metrics
│   ├── evaluator.py         # Evaluation harness
│   └── tinyml_metrics.py    # MCU deployment metrics
├── data/
│   ├── __init__.py
│   ├── transforms.py        # Z-score, gravity separation
│   ├── windowing.py         # Segmentation utilities
│   └── semantic_embeddings.py # Attribute/text embedding generation
├── utils/
│   ├── __init__.py
│   ├── quantization.py      # Weight and activation quantization
│   ├── pruning.py           # Structured pruning
│   ├── lut.py               # LUT generation for splines
│   └── export.py            # TFLM export utilities
├── experiments/
│   ├── __init__.py
│   ├── exp1_seen_classes.py  # Experiment 1: HAR on seen classes
│   ├── exp2_zsl.py          # Experiment 2: Zero-shot performance
│   ├── exp3_gamma.py        # Experiment 3: Calibration sensitivity
│   ├── exp4_significance.py # Experiment 4: Statistical significance
│   ├── exp5_ablation.py     # Experiment 5: Ablation studies
│   └── exp6_tinyml.py       # Experiment 6: TinyML deployment
├── scripts/
│   ├── download_data.py     # Download and extract datasets
│   ├── run_all_experiments.py # Execute full pipeline
│   └── reproduce_paper.py    # Main reproduction entry point
└── tests/
    ├── __init__.py
    ├── test_kan_layer.py
    ├── test_kan.py
    ├── test_zsl_module.py
    └── test_preprocessing.py
```

---

## Module Responsibilities

### datasets/
- `base.py`: Abstract class defining dataset interface (load, preprocess, split)
- `uci_har.py`: UCI HAR specific parsing, subject-wise split (21/4/9)
- `wisdm.py`: WISDM parsing, subject-wise split (35/8/8)
- `pamap2.py`: PAMAP2 parsing with heart-rate interpolation, subject split (6/1/2)
- `preprocessor.py`: Unified preprocessing pipeline (resample, normalize, window)

### models/
- `kan_layer.py`: Single KAN layer: W * x + b, then univariate B-spline functions per neuron
- `kan.py`: Stack of L KAN layers producing latent vector z ∈ R^{d_L}
- `classifier.py`: Linear head V * z + c for seen-class softmax
- `zsl_module.py`: Semantic projection W_sem, alignment loss, semantic softmax, cosine compatibility, calibration
- `explainer.py`: Gradient attributions, sensor/temporal aggregation, SHAP-style importance, spline visualization
- `baselines/`: kNN, SVM, RF (sklearn); 1D-CNN, LSTM, CNN-LSTM, Transformer (PyTorch)

### training/
- `trainer.py`: Mini-batch training loop with loss computation and gradient updates
- `optimizer.py`: Adam with configurable lr, weight decay, optional cosine annealing
- `losses.py`: CrossEntropyLoss, ZSL alignment loss, semantic CE loss, combined loss
- `early_stopping.py`: Monitor validation metric, patience-based stopping

### evaluation/
- `metrics.py`: Accuracy, precision, recall, F1 (macro/micro), ZSL accuracy, harmonic mean
- `evaluator.py`: Runs inference, computes metrics, handles ZSL/gZSL modes
- `tinyml_metrics.py`: Measures flash, RAM, latency, energy (or estimates from model)

### data/
- `transforms.py`: Gravity separation (FIR low-pass), z-score normalization
- `windowing.py`: Sliding window segmentation with configurable T and stride
- `semantic_embeddings.py`: Attribute encoding and text embedding generation

### utils/
- `quantization.py`: Symmetric uniform int8 quantization, per-tensor/layer scales
- `pruning.py`: Structured pruning (remove neurons/channels)
- `lut.py`: Pre-sample spline functions, quantize, store as lookup tables
- `export.py`: Export to ONNX, then to TFLM flatbuffer format

### experiments/
- `exp1_seen_classes.py`: Train and evaluate all models on seen classes
- `exp2_zsl.py`: Train TinyKAN-HAR with ZSL module, evaluate ZSL/gZSL
- `exp3_gamma.py`: Sweep gamma values on trained model
- `exp4_significance.py`: Multi-seed training and statistical testing
- `exp5_ablation.py`: Run all ablation variants from Table 7/8/9
- `exp6_tinyml.py`: Quantize, generate LUTs, export, measure deployment metrics

---

## Data Structures

### Input Tensor
- UCI HAR: `X ∈ R^{128 × 6}` (T=128 samples, D=6 channels)
- WISDM: `X ∈ R^{200 × 12}` (T=200 samples, D=12 channels)
- PAMAP2: `X ∈ R^{250 × 28}` (T=250 samples, D=28 channels)

### KAN Latent Vector
- `z ∈ R^{d_L}` where `d_L ∈ {64, 128, 256}` (default 128)

### Semantic Embedding
- `s_y ∈ R^m` where m depends on embedding type (attributes or text model)
- Hybrid: concatenation of attribute vector (fixed m_ATTR) and text embedding (fixed m_TEXT)

### Calibration Factor
- `γ ∈ [0.0, 1.0]` (default 0.5 selected on validation)

---

## Entry Points and Execution Flow

### Primary Entry: `scripts/reproduce_paper.py`
1. Load config.yaml
2. Download and preprocess datasets
3. For each experiment 1-6:
   - Load or build appropriate model
   - Run training if needed
   - Evaluate and record metrics
4. Generate comparison tables vs. paper results

### Experiment Execution Order
```
reproduce_paper.py
  ├── download_and_preprocess()
  ├── exp1_seen_classes()      → trains all models on seen classes
  │     └── (featurizers for kNN/SVM/RF + deep models)
  ├── exp2_zsl()               → trains TinyKAN-HAR+ZSL + baseline ZSL
  │     └── (depends on exp1 for base feature extractors)
  ├── exp3_gamma()             → uses exp2 model, sweeps gamma
  ├── exp4_significance()      → multi-seed exp1+exp2
  ├── exp5_ablation()          → trains ablation variants
  │     └── (depends on exp1 base model architecture)
  └── exp6_tinyml()            → quantizes, exports, measures
        └── (depends on exp1/exp5 for best model)
```

---

## External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0 | Deep learning framework |
| NumPy | ≥1.24 | Array operations |
| Scikit-learn | ≥1.3 | kNN, SVM, RF baselines |
| Pandas | ≥2.0 | Dataset handling |
| PyYAML | ≥6.0 | Config file parsing |
| TFLite Micro | (build) | TinyML deployment |
| CMSIS-NN | (build) | ARM Cortex-M optimized kernels |
| Matplotlib | ≥3.7 | Visualization (explanations) |
| Seaborn | ≥0.12 | Plotting attributions |
| SciPy | ≥1.10 | Statistical tests (t-test) |

---

## Traceability: Experiments → Modules

| Experiment | Modules Required |
|------------|-----------------|
| Exp 1 (Seen-class HAR) | datasets/*, models/kan.py, models/classifier.py, models/baselines/*, training/*, evaluation/* |
| Exp 2 (ZSL/gZSL) | + models/zsl_module.py, data/semantic_embeddings.py |
| Exp 3 (γ robustness) | + evaluation/evaluator.py (gamma sweep) |
| Exp 4 (Significance) | + training/trainer.py (multi-seed) |
| Exp 5 (Ablations) | + utils/quantization.py, utils/pruning.py, utils/lut.py |
| Exp 6 (TinyML) | + utils/export.py, evaluation/tinyml_metrics.py |

Every experiment from `plan.md` maps to at least one module. No experiment is orphaned.