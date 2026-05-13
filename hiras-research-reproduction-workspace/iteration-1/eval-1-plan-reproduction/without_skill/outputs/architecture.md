# Architecture: TinyKAN-HAR Reproduction

## 1. Directory Layout

```
D:\HiRAS\
├── paper.md                              # Source paper
├── hiras-research-reproduction-workspace/
│   └── iteration-1/
│       └── eval-1-plan-reproduction/
│           └── without_skill/
│               └── outputs/
│                   ├── plan.md           # High-level experiment plan
│                   ├── architecture.md    # This file
│                   ├── dependency.md     # Module dependencies
│                   └── config.yaml       # Hyperparameters
├── data/                                 # Dataset storage
│   ├── uci_har/                          # UCI HAR dataset
│   ├── wisdm/                            # WISDM dataset
│   └── pamap2/                           # PAMAP2 dataset
├── src/                                  # Source code
│   ├── __init__.py
│   ├── data/                             # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── uci_har.py                    # UCI HAR loader + preprocessing
│   │   ├── wisdm.py                       # WISDM loader + preprocessing
│   │   ├── pamap2.py                      # PAMAP2 loader + preprocessing
│   │   ├── preprocessing.py               # Shared preprocessing (resampling, normalization)
│   │   └── dataset.py                     # Unified dataset window + label handling
│   ├── kan/                              # KAN core implementation
│   │   ├── __init__.py
│   │   ├── spline.py                      # B-spline basis functions
│   │   ├── kan_layer.py                   # Single KAN layer (linear mixing + spline)
│   │   ├── kan_model.py                   # Multi-layer KAN (L=3 default)
│   │   └── regularization.py             # Weight decay, smoothness penalty, dropout
│   ├── zsl/                              # Zero-shot learning module
│   │   ├── __init__.py
│   │   ├── embeddings.py                  # Semantic embedding generation (ATTR, TEXT, HYBRID)
│   │   ├── projection.py                  # Semantic projection layer W_sem
│   │   ├── compatibility.py               # Cosine similarity compatibility function
│   │   └── calibration.py                 # Score calibration with gamma
│   ├── models/                           # All HAR models
│   │   ├── __init__.py
│   │   ├── tinykan_har.py                 # Full TinyKAN-HAR (KAN + ZSL + classification)
│   │   ├── baselines/                     # Baseline models
│   │   │   ├── __init__.py
│   │   │   ├── knn.py                     # k-Nearest Neighbors
│   │   │   ├── svm.py                     # SVM (RBF kernel)
│   │   │   ├── random_forest.py           # Random Forest
│   │   │   ├── cnn1d.py                   # 1D CNN
│   │   │   ├── lstm.py                    # LSTM
│   │   │   ├── cnn_lstm.py               # CNN-LSTM
│   │   │   └── transformer.py             # Transformer encoder
│   │   └── feature_extractor.py          # Hand-crafted features for classical baselines
│   ├── training/                         # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py                     # Main training loop
│   │   ├── optimizer.py                  # Adam optimizer with LR schedule
│   │   └── early_stopping.py             # Early stopping monitor
│   ├── evaluation/                        # Evaluation and metrics
│   │   ├── __init__.py
│   │   ├── classification_metrics.py     # Accuracy, precision, recall, F1
│   │   ├── zsl_metrics.py               # Acc_ZSL, Acc_seen, Acc_unseen, harmonic mean
│   │   ├── statistical_tests.py         # Paired t-test, multi-seed analysis
│   │   └── ablations.py                 # Ablation study runner
│   ├── explainability/                   # Explainability module
│   │   ├── __init__.py
│   │   ├── gradient_attribution.py      # Gradient-based attributions
│   │   ├── aggregation.py               # Sensor-level and temporal aggregation
│   │   ├── shap.py                      # SHAP-style global importance
│   │   └── visualization.py             # Spline function plots, attribution heatmaps
│   ├── deployment/                       # TinyML deployment
│   │   ├── __init__.py
│   │   ├── quantization.py             # 8-bit uniform quantization
│   │   ├── lut.py                       # LUT generation for spline evaluation
│   │   ├── pruning.py                   # Structured pruning
│   │   └── metrics.py                   # Flash, RAM, latency, energy estimation
│   └── utils/                           # Utilities
│       ├── __init__.py
│       ├── seeds.py                     # Random seed management
│       └── config.py                    # Config file loader
├── scripts/                             # Executable scripts
│   ├── train_kan_har.py                 # Train TinyKAN-HAR
│   ├── train_baselines.py               # Train all baseline models
│   ├── evaluate_all.py                  # Run full evaluation suite
│   ├── run_ablation.py                  # Run ablation studies
│   ├── generate_explanations.py         # Generate explanation figures
│   └── deploy_tinyml.py                 # Quantize, prune, estimate TinyML metrics
└── outputs/                             # Experiment outputs
    ├── models/                          # Saved model checkpoints
    ├── results/                         # Evaluation results (CSV/JSON)
    ├── figures/                         # Generated figures
    └── tinyml/                          # TinyML model artifacts
```

---

## 2. Module Responsibilities

### `src/data/`
- **uci_har.py**: Download URL, subject-wise train/val/test split (21/4/9), window=128, 50% overlap, label mapping
- **wisdm.py**: Download URL, subject split (35/8/8), window=200, 20Hz→50Hz resampling, 12 channels, label mapping
- **pamap2.py**: Download URL, subject split (6/1/2), window=250, 100Hz→50Hz downsampling, 28 channels, missing value interpolation, heart-rate handling
- **preprocessing.py**: Shared steps — temporal alignment, gravity separation (FIR low-pass), per-channel z-score normalization (compute µ, σ on train only)
- **dataset.py**: Unified `HARWindowDataset` class returning `(window_tensor, label_idx, subject_id)`; handles seen/unseen label partitioning

### `src/kan/`
- **spline.py**: B-spline basis functions B_k(u) with cubic splines; configurable knot grid over input range
- **kan_layer.py**: `KANLayer(l, d_in, d_out, K)` — pre-activation `u = Wx + b`, spline evaluation `φ_j(u_j) = Σ_k θ_{j,k} B_k(u_j)`, returns vector of same shape as pre-activation
- **kan_model.py**: `KANModel(L, d_0, d_1, ..., d_L, K, dropout)` — stacks L KAN layers; forward returns latent `z`
- **regularization.py**: `kan_regularization(model, λ_lin, λ_smooth)` — computes weight decay on W and V, smoothness penalty on spline coefficients θ

### `src/zsl/`
- **embeddings.py**: `SemanticEmbeddings` class — supports ATTR (binary attributes), TEXT (text→embedding via pretrained LM), HYBRID (concat); loads/creates for each activity label
- **projection.py**: `SemanticProjection(d_L, m)` — linear layer `h = W_sem @ z` mapping latent to semantic space
- **compatibility.py**: `cosine_compatibility(h, s_y)` — cosine similarity between projected latent and semantic embedding
- **calibration.py**: `calibrate_scores(g_phi, y, γ)` — subtracts γ from seen-class scores

### `src/models/`
- **tinykan_har.py**: `TinyKANHAR` — KAN backbone + semantic projection + classification head; forward: z = kan(x), h = W_sem @ z, g = cosine(h, S), logits = V @ z + c; training loss = CE + λ_ZSL*(λ_align*L_align + λ_semCE*L_semCE)
- **baselines/knn.py**: kNN classifier with k tuned on validation; Euclidean distance on feature vectors
- **baselines/svm.py**: SVM with RBF kernel; C and γ tuned via grid search
- **baselines/random_forest.py**: Random Forest ensemble; n_estimators, max_features tuned
- **baselines/cnn1d.py**: 1D CNN — 3 conv layers with BatchNorm, ReLU, pooling; FC classification head
- **baselines/lstm.py**: LSTM — 1 or 2 LSTM layers, optional CNN front-end, pooled hidden state
- **baselines/cnn_lstm.py**: CNN front-end + LSTM; concat final hidden states or use attention pooling
- **baselines/transformer.py**: Transformer encoder — positional encoding, multi-head self-attention, FFN, class token or mean pooling
- **baselines/zsl_head.py**: Attaches ZSL module (semantic projection + cosine compatibility) to any backbone
- **feature_extractor.py**: `extract_features(window)` — mean, std, RMS, SMA, dominant frequency, spectral entropy per channel; concatenates into feature vector

### `src/training/`
- **trainer.py**: `Trainer` class — fit(model, train_loader, val_loader, epochs, device); implements training loop, backprop, metric tracking, checkpointing
- **optimizer.py**: `create_optimizer(model, lr, weight_decay)` returning Adam; `create_scheduler(optimizer, schedule_type, ...)` for step or cosine annealing
- **early_stopping.py**: `EarlyStopping(patience, metric)` — monitors validation metric, restores best model, stops if no improvement for `patience` epochs

### `src/evaluation/`
- **classification_metrics.py**: `compute_metrics(y_true, y_pred)` — accuracy, precision (per-class + macro), recall (per-class + macro), F1 (per-class + macro/micro)
- **zsl_metrics.py**: `compute_zsl_metrics(model, seen_loader, unseen_loader, semantic_embs, γ)` — pure ZSL accuracy, gZSL Acc_seen, Acc_unseen, harmonic mean H
- **statistical_tests.py**: `multi_seed_analysis(model_fn, seeds, ...)` — runs 5 seeds, computes mean±std, paired t-test
- **ablations.py**: `run_ablation(config)` — iterates over ablation variants defined in config, logs all metrics

### `src/explainability/`
- **gradient_attribution.py**: `gradient_attribution(model, x, target_class)` — grad × input method; returns T×D attribution matrix
- **aggregation.py**: `aggregate_sensor(attribution_matrix)` — sums over time per channel; `aggregate_temporal(attribution_matrix)` — sums over channels per time step
- **shap.py**: `shap_importance(model, X_train, n_samples)` — sampling-based SHAP approximation; returns per-channel global importance
- **visualization.py**: `plot_spline_functions(model, ...)` — plots learned univariate functions per layer; `plot_attribution_heatmap(attr_matrix, ...)` — time×sensor heatmap

### `src/deployment/`
- **quantization.py**: `quantize_model(model, bit_width=8)` — symmetric uniform quantization; returns quantized model with scale factors
- **lut.py**: `generate_lut(model, bit_width=8)` — pre-samples each spline function on a uniform grid; stores as fixed-point LUT arrays
- **pruning.py**: `structured_prune(model, rate)` — removes entire rows/columns from weight matrices
- **metrics.py**: `estimate_tinyml_metrics(model, quantized)` — computes flash size (kB), peak RAM (kB), latency (ms), energy (µJ)

---

## 3. Data Flow

```
Raw sensor files (UCI/WISDM/PAMAP2)
    │
    ▼
data/preprocessing.py ──► Unified HARWindowDataset
    │                        │
    │                   ┌────┴──────────────────┐
    │                   │                         │
    ▼                   ▼                         ▼
kan/kan_model.py   zsl/projection.py       baselines/*.py
(KAN backbone)    (semantic projection)   (classical + deep)
    │                   │                         │
    │                   │                         │
    └───────┬───────────┴─────────────────────────┘
            │
            ▼
    TinyKANHAR (full model)  OR  baseline model
            │
            ▼
    trainer.py (training loop)
            │
            ▼
    Evaluation (classification + ZSL metrics)
            │
            ├──► evaluation/classification_metrics.py
            ├──► evaluation/zsl_metrics.py
            ├──► evaluation/ablations.py
            ├──► evaluation/statistical_tests.py
            └──► explainability/visualization.py

    TinyML Deployment:
            │
            ▼
    deployment/quantization.py ──► deployment/lut.py ──► deployment/pruning.py
                                    │
                                    ▼
                              deployment/metrics.py
                              (flash/RAM/latency/energy)
```

---

## 4. Interfaces and Key Classes

### `HARWindowDataset`
```python
class HARWindowDataset(Dataset):
    def __init__(self, windows, labels, label_to_idx, subject_ids)
    def __getitem__(self, idx) -> Tuple[Tensor, int, int]:  # window, label, subject
    def __len__(self) -> int
    @property
    def seen_mask(self) -> BoolTensor   # windows with seen labels
    @property
    def unseen_mask(self) -> BoolTensor # windows with unseen labels
```

### `KANLayer`
```python
class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_knots: int = 5)
    def forward(self, x: Tensor) -> Tensor  # shape: (batch, out_features)
    def spline_evaluate(self, u: Tensor) -> Tensor  # evaluate univariate splines
    @property
    def num_params(self) -> int
```

### `TinyKANHAR`
```python
class TinyKANHAR(nn.Module):
    def __init__(self, d_in: int, L: int, d_hidden: List[int], d_latent: int,
                 num_knots: int, num_seen_classes: int, semantic_dim: int,
                 dropout: float = 0.1)
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # returns: (logits, semantic_scores, z) for seen classes
    def zero_shot_forward(self, x: Tensor) -> Tensor:
        # returns: compatibility scores g_phi for all classes (seen + unseen)
    def project_to_semantic(self, z: Tensor) -> Tensor:
```

### `Trainer`
```python
class Trainer:
    def __init__(self, model, optimizer, scheduler, device, config)
    def train(self, train_loader, val_loader, epochs) -> Dict:
        # returns: {'best_epoch': int, 'best_val_acc': float, 'train_history': [...]}
    def evaluate(self, test_loader) -> Dict
```

---

## 5. Entry Points

| Script | Purpose |
|---|---|
| `scripts/train_kan_har.py` | Train TinyKAN-HAR on one dataset with given config |
| `scripts/train_baselines.py` | Train all baseline models (kNN, SVM, RF, CNN, LSTM, CNN-LSTM, Transformer) |
| `scripts/evaluate_all.py` | Run full evaluation: seen-class accuracy, ZSL, gZSL, TinyML metrics |
| `scripts/run_ablation.py` | Run all ablation variants; save results to CSV |
| `scripts/generate_explanations.py` | Generate attribution heatmaps, spline plots for case studies |
| `scripts/deploy_tinyml.py` | Quantize, generate LUTs, prune, estimate TinyML metrics |

---

## 6. External Dependencies

| Package | Version | Purpose |
|---|---|---|
| torch | ≥2.0 | Deep learning framework |
| numpy | ≥1.24 | Numerical computing |
| scipy | ≥1.11 | Signal processing (low-pass filter), interpolation |
| scikit-learn | ≥1.3 | kNN, SVM, RF baselines; metrics |
| matplotlib | ≥3.7 | Visualization |
| pandas | ≥2.0 | Result logging |
| tqdm | ≥4.65 | Progress bars |
| pyyaml | ≥6.0 | Config file loading |
| requests | ≥2.31 | Dataset download (optional) |

---

## 7. Experiment-to-Module Traceability

| Experiment | Responsible Modules |
|---|---|
| 1. HAR on seen classes | `data/`, `kan/`, `models/tinykan_har.py`, `models/baselines/`, `training/`, `evaluation/classification_metrics.py` |
| 2. ZSL + gZSL | `zsl/`, `models/tinykan_har.py` (zero_shot_forward), `evaluation/zsl_metrics.py` |
| 3. γ robustness | `zsl/calibration.py`, `evaluation/zsl_metrics.py` (sweep γ) |
| 4. Statistical significance | `evaluation/statistical_tests.py`, `training/trainer.py` (multi-seed) |
| 5. Ablations | `evaluation/ablations.py`, all model components |
| 6. TinyML deployment | `deployment/quantization.py`, `deployment/lut.py`, `deployment/pruning.py`, `deployment/metrics.py` |
| 7. KAN depth/latent | `kan/kan_model.py` (L, dL config), `evaluation/ablations.py` |
| 8. Explanations | `explainability/` all modules, `evaluation/visualization.py` |