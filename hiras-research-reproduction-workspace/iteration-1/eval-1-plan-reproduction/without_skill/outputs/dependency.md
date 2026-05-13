# Dependency Analysis: TinyKAN-HAR Reproduction

## 1. Package Requirements (requirements.txt)

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0.0
requests>=2.31.0
```

---

## 2. Inter-File Dependencies (Import Graph)

### Tier 0 — No dependencies (must implement first)
```
src/data/spline.py
src/utils/seeds.py
src/utils/config.py
```

### Tier 1 — Depends only on Tier 0
```
src/kan/kan_layer.py        imports: src/data/spline.py
src/data/preprocessing.py   imports: scipy (signal), numpy
src/data/dataset.py         imports: torch.utils.data, numpy
```

### Tier 2 — Depends on Tier 1
```
src/kan/kan_model.py        imports: src/kan/kan_layer.py, torch.nn
src/kan/regularization.py   imports: torch, src/kan/kan_layer.py
src/zsl/embeddings.py       imports: numpy
src/zsl/projection.py       imports: torch.nn
src/zsl/compatibility.py    imports: torch.nn.functional
src/zsl/calibration.py      imports: numpy
```

### Tier 3 — Depends on Tier 2
```
src/models/tinykan_har.py   imports: torch.nn, src/kan/kan_model.py, src/zsl/projection.py, src/zsl/compatibility.py, src/zsl/calibration.py
src/models/baselines/cnn1d.py  imports: torch.nn
src/models/baselines/lstm.py   imports: torch.nn
src/models/baselines/transformer.py  imports: torch.nn, torch.nn.functional
src/models/feature_extractor.py  imports: scipy.signal, numpy
```

### Tier 4 — Depends on Tier 3
```
src/training/trainer.py    imports: torch.optim, torch.nn, src/models/tinykan_har.py
src/training/optimizer.py   imports: torch.optim
src/training/early_stopping.py  imports: numpy
src/evaluation/classification_metrics.py  imports: sklearn.metrics, numpy
src/evaluation/zsl_metrics.py  imports: torch, numpy
src/evaluation/statistical_tests.py  imports: scipy.stats, numpy
src/explainability/gradient_attribution.py  imports: torch.autograd
src/explainability/shap.py  imports: torch, numpy, tqdm
src/deployment/quantization.py  imports: torch.nn, torch.quantization
src/deployment/lut.py       imports: torch, numpy
src/deployment/pruning.py   imports: torch.nn
src/deployment/metrics.py   imports: torch, numpy
```

### Tier 5 — Depends on Tier 4
```
src/training/__init__.py    imports: trainer.py, optimizer.py, early_stopping.py
src/evaluation/__init__.py  imports: classification_metrics.py, zsl_metrics.py, statistical_tests.py, ablations.py
src/explainability/__init__.py  imports: gradient_attribution.py, aggregation.py, shap.py, visualization.py
src/deployment/__init__.py  imports: quantization.py, lut.py, pruning.py, metrics.py
src/models/__init__.py      imports: tinykan_har.py, baselines/*, feature_extractor.py
```

### Tier 6 — Top-level entry points
```
scripts/train_kan_har.py    imports: src.data.dataset, src.models.tinykan_har, src.training.trainer, src.utils.config, src.utils.seeds
scripts/train_baselines.py  imports: src.data.dataset, src.models.baselines.*, src.training.trainer, src.utils.config
scripts/evaluate_all.py     imports: src.evaluation.*, src.data.dataset, src.models.tinykan_har, src.models.baselines
scripts/run_ablation.py     imports: src.evaluation.ablations, src.models.tinykan_har, src.utils.config
scripts/generate_explanations.py  imports: src.explainability.*, src.data.dataset, src.models.tinykan_har
scripts/deploy_tinyml.py    imports: src.deployment.*, src.models.tinykan_har
```

---

## 3. Logic Analysis: Components, Classes, Functions

### 3.1 Data Layer

#### `src/data/spline.py`
- **Purpose:** B-spline basis function evaluation for KAN layers
- **Classes/Functions:**
  - `BSplineBasis(knots: np.ndarray, degree: int = 3)` — manages knot vector and basis function evaluation
  - `evaluate_basis(u: float, idx: int) -> float` — evaluate B_k(u) at scalar input
  - `evaluate_all_basis(u: float) -> np.ndarray` — returns all K basis function values at u
  - `spline_interpolation(u: float, coefficients: np.ndarray) -> float` — Σ θ_k * B_k(u)
- **Edge cases:** u outside knot range → return 0 or clamp; degree must be ≥1

#### `src/data/preprocessing.py`
- **Purpose:** Shared preprocessing pipeline across all datasets
- **Functions:**
  - `resample_to_target(signal: np.ndarray, orig_freq: float, target_freq: float) -> np.ndarray` — linear interpolation or low-pass filter + decimation
  - `separate_gravity(acc_signal: np.ndarray, fs: float, cutoff: float = 0.3) -> (np.ndarray, np.ndarray)` — FIR low-pass filter to extract gravity, compute body acceleration
  - `normalize_channel(windows: np.ndarray, mu: float, sigma: float, eps: float = 1e-8) -> np.ndarray` — z-score normalization
  - `compute_normalization_stats(windows: np.ndarray) -> (mu, sigma)` — per-channel mean and std over training set
  - `interpolate_missing(signal: np.ndarray, mask: np.ndarray) -> np.ndarray` — linear interpolation for gaps
- **Edge cases:** PAMAP2 heart-rate at 9Hz must be upsampled/interpolated to 50Hz; long missing gaps → discard windows

#### `src/data/dataset.py`
- **Purpose:** Unified dataset interface for all three HAR datasets
- **Classes:**
  - `HARWindowDataset(torch.utils.data.Dataset)` — holds windows, labels, subject IDs; supports seen/unseen masking
    - `__init__(windows, labels, label_to_idx, subject_ids)` — store preprocessed windows and mappings
    - `__getitem__(idx) -> (window_tensor, label_idx, subject_id)`
    - `__len__() -> int`
    - `seen_indices() -> List[int]` — indices of windows with seen labels
    - `unseen_indices() -> List[int]` — indices of windows with unseen labels
    - `split_by_subject(train_subjects, val_subjects, test_subjects) -> (train_ds, val_ds, test_ds)`

---

### 3.2 KAN Layer

#### `src/kan/kan_layer.py`
- **Purpose:** Single KAN layer with linear mixing + univariate spline nonlinearity
- **Class:** `KANLayer(nn.Module)`
  - `__init__(in_features, out_features, num_knots=5, init_range=(-1, 1))`
    - `W: nn.Parameter (out_features × in_features)` — linear mixing matrix
    - `b: nn.Parameter (out_features)` — bias vector
    - `theta: nn.Parameter (out_features × num_knots)` — spline coefficients
    - Internal B-spline basis grid over init_range
  - `forward(x: Tensor) -> Tensor` — x ∈ R^{batch, in_features}, returns ∈ R^{batch, out_features}
    1. `u = x @ W.T + b`  (pre-activation)
    2. For each neuron j: `φ_j(u_j) = Σ_k theta[j,k] * B_k(u_j)`
    3. Return [φ_1(u_1), ..., φ_dout(u_dout)]
  - `init_as_identity()` — set theta so φ_j(u) ≈ u for u ∈ init_range
  - `get_spline_function(j) -> callable` — return function u → φ_j(u) for visualization

#### `src/kan/kan_model.py`
- **Purpose:** Multi-layer KAN (stacked KANLayers)
- **Class:** `KANModel(nn.Module)`
  - `__init__(layer_dims: List[int], num_knots: int = 5, dropout: float = 0.1)`
    - `layers: nn.ModuleList[KANLayer]` — one per hidden layer
    - `dropout: nn.Dropout`
  - `forward(x: Tensor) -> Tensor` — x ∈ R^{batch, d_0}, returns z ∈ R^{batch, d_L}
  - `get_spline_functions(layer_idx) -> List[callable]` — for explainability

#### `src/kan/regularization.py`
- **Purpose:** Compute regularization losses for KAN training
- **Functions:**
  - `weight_decay_loss(model) -> Tensor` — sum of ||W||_F^2 for all KAN layers and classification head
  - `spline_smoothness_loss(model) -> Tensor` — Σ_{l,j} Σ_{k=2}^{K-1} (θ_{j,k+1} - 2θ_{j,k} + θ_{j,k-1})^2
  - `kan_regularization(model, lambda_lin, lambda_smooth) -> (total_loss, lin_loss, smooth_loss)`
  - `apply_dropout(model, x, p_drop) -> Tensor` — stochastic masking during training

---

### 3.3 ZSL Module

#### `src/zsl/embeddings.py`
- **Purpose:** Generate semantic embeddings for activity labels
- **Class:** `SemanticEmbeddings`
  - `__init__(embedding_type: str, labels: List[str])` — type ∈ {ATTR, TEXT, HYBRID}
  - `create_attribute_embeddings(labels) -> np.ndarray` — manually defined binary attributes per activity (e.g., "locomotion", "upper_body", "high_intensity")
  - `create_text_embeddings(labels) -> np.ndarray` — use pretrained sentence transformer (e.g., sentence-transformers/all-MiniLM-L6-v2) to encode activity names/descriptions
  - `create_hybrid_embeddings(labels) -> np.ndarray` — concat of attribute + text embeddings
  - `get_embedding(label_idx) -> np.ndarray` — returns m-dimensional embedding for given label index
  - `get_all_embeddings() -> np.ndarray` — returns matrix S ∈ R^{m × |Y|}
  - `embedding_dim() -> int`

#### `src/zsl/projection.py`
- **Purpose:** Learnable projection from KAN latent space to semantic space
- **Class:** `SemanticProjection(nn.Module)`
  - `__init__(latent_dim: int, semantic_dim: int)`
    - `W_sem: nn.Parameter (semantic_dim × latent_dim)`
    - `reset_parameters()` — initialize near zero
  - `forward(z: Tensor) -> Tensor` — h = W_sem @ z ∈ R^{semantic_dim}
  - `get_projection_matrix() -> np.ndarray`

#### `src/zsl/compatibility.py`
- **Purpose:** Cosine similarity between projected latent and semantic embedding
- **Functions:**
  - `cosine_similarity(h: Tensor, s: Tensor) -> Tensor` — (h· s) / (||h||·||s||)
  - `compatibility_scores(z: Tensor, W_sem, S) -> Tensor` — for each label y: g_φ(z, s_y) = cos(W_sem @ z, s_y)
  - `semantic_softmax(scores: Tensor, temperature: float) -> Tensor` — softmax over compatibility scores

#### `src/zsl/calibration.py`
- **Purpose:** Score calibration for generalized ZSL
- **Functions:**
  - `calibrate(g_phi: Tensor, y_indices: List[int], gamma: float, is_seen: List[bool]) -> Tensor`
    - For seen labels: subtract γ; for unseen: leave unchanged
  - `find_optimal_gamma(model, val_loader, seen_indices, unseen_indices, search_range) -> float`

---

### 3.4 Models

#### `src/models/tinykan_har.py`
- **Purpose:** Full TinyKAN-HAR model: KAN + ZSL + classification
- **Class:** `TinyKANHAR(nn.Module)`
  - `__init__(d_in, L, d_hidden, d_latent, num_knots, num_seen_classes, semantic_dim, dropout=0.1)`
    - `kan: KANModel` — feature extractor
    - `semantic_proj: SemanticProjection`
    - `classifier: nn.Linear` — V @ z + c over seen classes
  - `forward(x: Tensor) -> (logits, compat_scores, z)`
    1. `z = kan(x)` — latent representation
    2. `h = semantic_proj(z)` — semantic projection
    3. `g = cosine(h, S)` — compatibility scores for all classes
    4. `logits = classifier(z)` — seen-class logits
    5. Return (logits, g, z)
  - `zero_shot_forward(x: Tensor, gamma: float) -> (calibrated_scores, predicted_class)`
    1. Compute g for all classes (seen + unseen)
    2. Apply calibration: g_seen -= gamma
    3. Return (calibrated_g, argmax(calibrated_g))
  - `compute_loss(x, y_seen, λ_ZSL, λ_align, λ_semCE, S) -> (total_loss, ce_loss, align_loss, sem_ce_loss)`

#### `src/models/baselines/`
- **knn.py:** `class KNNBaseline` — kNN with Euclidean distance, k tuned via validation
- **svm.py:** `class SVMBaseline` — SVM with RBF kernel, C/γ tuned via grid search
- **random_forest.py:** `class RandomForestBaseline` — ensemble of decision trees, n_estimators tuned
- **cnn1d.py:** `class CNN1D(nn.Module)` — 3 conv blocks (Conv1D, BatchNorm, ReLU, MaxPool), FC head
- **lstm.py:** `class LSTMHAR(nn.Module)` — optional CNN front-end + LSTM layers + FC head
- **cnn_lstm.py:** `class CNNLSTM(nn.Module)` — CNN front-end + bidirectional LSTM + FC head
- **transformer.py:** `class TransformerHAR(nn.Module)` — learnable positional encoding + Transformer encoder + FC head

#### `src/models/feature_extractor.py`
- **Functions:**
  - `extract_handcrafted_features(window: np.ndarray, fs: float) -> np.ndarray`
    - Per channel: mean, std, RMS, SMA, zero-crossing rate, dominant frequency (FFT), spectral entropy
    - Concatenate → feature vector
  - `extract_all_features(dataset: HARWindowDataset) -> (X, y)` — for classical ML baselines

---

### 3.5 Training

#### `src/training/trainer.py`
- **Class:** `Trainer`
  - `__init__(model, optimizer, scheduler, device, config)`
  - `train(train_ds, val_ds, epochs) -> Dict` — training loop with early stopping
    - For each epoch: forward pass, compute losses, backprop, update parameters
    - Validate after each epoch; track best model by val metric
    - Return history: {epoch: {train_loss, train_acc, val_loss, val_acc, ...}}
  - `evaluate(test_ds) -> Dict` — evaluate on test set, return metrics
  - `save_checkpoint(path)` / `load_checkpoint(path)`

#### `src/training/optimizer.py`
- **Functions:**
  - `create_adam_optimizer(model, lr, weight_decay) -> torch.optim.Adam`
  - `create_scheduler(optimizer, schedule_type, step_size, gamma, T_max)` — step or cosine annealing

#### `src/training/early_stopping.py`
- **Class:** `EarlyStopping`
  - `__init__(patience, min_delta, mode)`
  - `__call__(val_metric) -> bool` — returns True if should stop
  - `best_score` / `counter` — tracking

---

### 3.6 Evaluation

#### `src/evaluation/classification_metrics.py`
- **Functions:**
  - `accuracy(y_true, y_pred) -> float`
  - `precision_recall_f1(y_true, y_pred, average) -> Dict` — supports 'micro', 'macro', 'per-class'
  - `classification_report(y_true, y_pred, label_names) -> str`

#### `src/evaluation/zsl_metrics.py`
- **Functions:**
  - `pure_zsl_accuracy(model, unseen_loader, semantic_embs, device) -> float` — model predicts among unseen only; compare to ground truth
  - `gzsl_metrics(model, seen_loader, unseen_loader, semantic_embs, gamma, device) -> Dict` — returns {Acc_seen, Acc_unseen, H}
  - `harmonic_mean(acc_seen, acc_unseen) -> float`

#### `src/evaluation/statistical_tests.py`
- **Functions:**
  - `multi_seed_training(model_fn, dataset, seeds, config) -> List[Dict]` — runs model_fn with different seeds, returns list of metric dicts per seed
  - `paired_ttest(results1, results2, metric_name) -> (t_stat, p_value)`
  - `mean_std_report(results) -> (mean, std)`

#### `src/evaluation/ablations.py`
- **Class:** `AblationRunner`
  - `__init__(base_model, config)`
  - `run_all() -> pd.DataFrame` — iterates over ablation variants in config, runs evaluation, returns results table
  - `run_single(variant_name, variant_config) -> Dict`

---

### 3.7 Explainability

#### `src/explainability/gradient_attribution.py`
- **Functions:**
  - `gradient_x_input(model, x, target_class) -> np.ndarray` — compute ∂score/∂x * x; shape T×D
  - `saliency_map(model, x, target_class) -> np.ndarray` — absolute gradients

#### `src/explainability/aggregation.py`
- **Functions:**
  - `aggregate_sensor(attribution_matrix) -> np.ndarray` — sum over time axis → D relevance scores
  - `aggregate_temporal(attribution_matrix) -> np.ndarray` — sum over sensor axis → T relevance scores
  - `group_by_device(relevance_per_channel, channel_to_device_map) -> Dict[device, score]` — for WISDM phone+watch

#### `src/explainability/shap.py`
- **Functions:**
  - `sampling_shap(model, x_train, x_test, n_samples) -> np.ndarray` — sampling-based SHAP approximation; returns importance per channel

#### `src/explainability/visualization.py`
- **Functions:**
  - `plot_spline_functions(model, layer_idx, neurons, save_path)` — plot φ_j(u) curves for selected neurons
  - `plot_attribution_heatmap(attr_matrix, sensor_names, time_axis, save_path)` — T×D heatmap
  - `plot_sensor_relevance(relevance_scores, sensor_names, save_path)` — bar plot
  - `plot_temporal_relevance(relevance_curve, time_axis, save_path)` — line plot
  - `plot_class_activations(model, class_names, layer_idx, neurons, save_path)` — mean activations per class

---

### 3.8 Deployment

#### `src/deployment/quantization.py`
- **Functions:**
  - `symmetric_quantize(tensor, bit_width) -> (quantized_tensor, scale)` — uniform symmetric quantization
  - `quantize_model(model, bit_width=8)` — replace all Float weights with int8, add scale factors
  - `quantize_activations(tensor, bit_width) -> (quantized_tensor, scale)`

#### `src/deployment/lut.py`
- **Functions:**
  - `generate_spline_lut(kan_model, bit_width, grid_size)` — pre-sample each univariate spline on uniform grid over input range; store as intN lookup table
  - `interpolate_lut(u, lut_grid, lut_values) -> float` — linear interpolation between LUT entries
  - `export_lut_c_array(lut_data) -> str` — export LUT as C array for TinyML deployment

#### `src/deployment/pruning.py`
- **Functions:**
  - `structured_prune_layer(layer, rate) -> nn.Parameter` — zero out entire rows/columns of W
  - `structured_prune_model(model, pruning_rates)` — apply per-layer pruning rates
  - `compute_sparsity(model) -> float` — fraction of zero weights

#### `src/deployment/metrics.py`
- **Functions:**
  - `count_parameters(model, quantized) -> int` — total trainable parameters
  - `estimate_flash_size(model, quantized, bit_width) -> float` — in kilobytes
  - `estimate_ram_size(model, batch_size, quantized, bit_width) -> float` — peak activation buffers in kB
  - `estimate_latency(model, hardware_profile) -> float` — in milliseconds
  - `estimate_energy(latency, power_draw) -> float` — in microjoules

---

## 4. Task Ordering (Topological Sort)

```
Step 1: Implement data layer
  → src/utils/config.py
  → src/utils/seeds.py
  → src/data/spline.py
  → src/data/preprocessing.py
  → src/data/dataset.py

Step 2: Implement KAN core
  → src/kan/kan_layer.py
  → src/kan/kan_model.py
  → src/kan/regularization.py

Step 3: Implement ZSL components
  → src/zsl/embeddings.py
  → src/zsl/projection.py
  → src/zsl/compatibility.py
  → src/zsl/calibration.py

Step 4: Implement models
  → src/models/tinykan_har.py
  → src/models/baselines/cnn1d.py
  → src/models/baselines/lstm.py
  → src/models/baselines/transformer.py
  → src/models/feature_extractor.py
  → src/models/baselines/knn.py, svm.py, random_forest.py
  → src/models/baselines/cnn_lstm.py

Step 5: Implement training
  → src/training/optimizer.py
  → src/training/early_stopping.py
  → src/training/trainer.py

Step 6: Implement evaluation
  → src/evaluation/classification_metrics.py
  → src/evaluation/zsl_metrics.py
  → src/evaluation/statistical_tests.py
  → src/evaluation/ablations.py

Step 7: Implement explainability
  → src/explainability/gradient_attribution.py
  → src/explainability/aggregation.py
  → src/explainability/shap.py
  → src/explainability/visualization.py

Step 8: Implement deployment
  → src/deployment/quantization.py
  → src/deployment/lut.py
  → src/deployment/pruning.py
  → src/deployment/metrics.py

Step 9: Implement entry point scripts
  → scripts/train_kan_har.py
  → scripts/train_baselines.py
  → scripts/evaluate_all.py
  → scripts/run_ablation.py
  → scripts/generate_explanations.py
  → scripts/deploy_tinyml.py
```

---

## 5. Circular Import Check

All modules import from strictly lower-tier modules. No circular imports detected.
- `src/kan/` imports from `src/data/spline.py` only
- `src/zsl/` imports from `torch`, `numpy` (no local cross-imports)
- `src/models/tinykan_har.py` imports from `kan`, `zsl` — both are lower tier
- `src/training/` imports from `models` — which is higher tier than data/kan/zsl; no cycles
- No module imports from `evaluation/` or `explainability/` except top-level scripts