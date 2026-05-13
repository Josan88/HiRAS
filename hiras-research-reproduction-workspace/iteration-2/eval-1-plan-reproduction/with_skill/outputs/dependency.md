# Phase 3: Dependency Modelling — TinyKAN-HAR Reproduction

## Package Dependencies (requirements.txt)

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0.0
requests>=2.28.0          # For dataset download
appdirs>=1.4.4             # For dataset download
```

**Note on TinyML dependencies:** Full TinyML deployment requires TensorFlow Lite Micro and CMSIS-NN, which must be built from source for the target ARM Cortex-M toolchain. MCU profiling additionally requires either physical hardware or QEMU emulation. These are **not** pip-installable and must be set up separately.

---

## Inter-File Dependencies

```
preprocessing/dataset.py
    └── imports: numpy, torch.utils.data
    └── used by: all training experiments (1, 2, 4, 5, 8)

preprocessing/temporal_align.py
    └── imports: scipy.signal.resample_poly
    └── used by: preprocessing/dataset.py

preprocessing/gravity_separation.py
    └── imports: scipy.signal.firwin, lfilter
    └── used by: preprocessing/dataset.py

preprocessing/normalization.py
    └── imports: numpy
    └── used by: preprocessing/dataset.py

preprocessing/missing_values.py
    └── imports: numpy, scipy.interpolate.interp1d
    └── used by: preprocessing/dataset.py

models/kan_layer.py
    └── imports: torch.nn, torch
    └── exports: KANLayer
    └── used by: models/kan.py

models/kan.py
    └── imports: torch.nn, models.kan_layer.KANLayer
    └── exports: KAN
    └── used by: models/tinykan_har.py, experiments/exp8_depth_width.py

models/classification_head.py
    └── imports: torch.nn
    └── exports: ClassificationHead
    └── used by: models/tinykan_har.py

models/zsl_module.py
    └── imports: torch.nn, torch
    └── exports: ZSLModule
    └── used by: models/tinykan_har.py, experiments/exp2_zsl.py, exp3_gamma_robustness.py

models/tinykan_har.py
    └── imports: torch.nn, models.kan.KAN, models.classification_head.ClassificationHead, models.zsl_module.ZSLModule
    └── exports: TinyKANHAR
    └── used by: training/train_kan.py, experiments/exp1_har_performance.py, exp2_zsl.py, exp5_ablations.py

models/baselines/knn.py
    └── imports: sklearn.neighbors.KNeighborsClassifier
    └── used by: experiments/exp1_har_performance.py

models/baselines/svm.py
    └── imports: sklearn.svm.SVC
    └── used by: experiments/exp1_har_performance.py

models/baselines/random_forest.py
    └── imports: sklearn.ensemble.RandomForestClassifier
    └── used by: experiments/exp1_har_performance.py

models/baselines/cnn1d.py
    └── imports: torch.nn
    └── used by: experiments/exp1_har_performance.py

models/baselines/lstm.py
    └── imports: torch.nn
    └── used by: experiments/exp1_har_performance.py

models/baselines/cnn_lstm.py
    └── imports: torch.nn
    └── used by: experiments/exp1_har_performance.py

models/baselines/transformer.py
    └── imports: torch.nn
    └── used by: experiments/exp1_har_performance.py

training/losses.py
    └── imports: torch.nn.functional
    └── exports: cross_entropy_loss, alignment_loss, sem_ce_loss, smoothness_loss, total_loss
    └── used by: training/train_kan.py

training/train_kan.py
    └── imports: torch.optim.Adam, torch.utils.data.DataLoader, training.losses, evaluation.har_metrics, evaluation.zsl_metrics
    └── used by: experiments/exp1_har_performance.py, exp2_zsl.py, exp5_ablations.py, exp8_depth_width.py

training/multi_seed.py
    └── imports: training.train_kan
    └── used by: experiments/exp4_statistical.py

evaluation/har_metrics.py
    └── imports: sklearn.metrics
    └── used by: training/train_kan.py, experiments/exp1_har_performance.py

evaluation/zsl_metrics.py
    └── imports: torch
    └── used by: training/train_kan.py, experiments/exp2_zsl.py, exp3_gamma_robustness.py

evaluation/statistical_tests.py
    └── imports: scipy.stats.ttest_rel
    └── used by: experiments/exp4_statistical.py

evaluation/ablations.py
    └── used by: experiments/exp5_ablations.py

explainability/gradient_attribution.py
    └── imports: torch.autograd
    └── used by: experiments/exp7_viz.py

explainability/sensor_aggregation.py
    └── imports: numpy
    └── used by: experiments/exp7_viz.py

explainability/temporal_aggregation.py
    └── imports: numpy
    └── used by: experiments/exp7_viz.py

explainability/shap_approx.py
    └── imports: torch, numpy
    └── used by: experiments/exp7_viz.py

explainability/spline_viz.py
    └── imports: matplotlib.pyplot, numpy
    └── used by: experiments/exp7_viz.py

deployment/quantize.py
    └── imports: torch
    └── used by: experiments/exp5_ablations.py, exp6_tinyml.py

deployment/lut_generator.py
    └── imports: numpy, torch
    └── used by: experiments/exp5_ablations.py, exp6_tinyml.py

deployment/tflm_converter.py
    └── imports: subprocess (for xxd, etc.), torch
    └── used by: experiments/exp6_tinyml.py

deployment/mcu_profiler.py
    └── imports: subprocess (for QEMU or remote probe)
    └── used by: experiments/exp5_ablations.py, exp6_tinyml.py
```

---

## Task Ordering (Topological Sort)

**Phase A: Data Pipeline (must be first)**
1. `preprocessing/temporal_align.py` — no dependencies
2. `preprocessing/gravity_separation.py` — no dependencies
3. `preprocessing/normalization.py` — no dependencies
4. `preprocessing/missing_values.py` — no dependencies
5. `preprocessing/segmentation.py` — no dependencies
6. `preprocessing/dataset.py` — depends on all above

**Phase B: Model Core**
7. `models/kan_layer.py` — no dependencies
8. `models/kan.py` — depends on kan_layer.py
9. `models/classification_head.py` — no dependencies
10. `models/zsl_module.py` — no dependencies
11. `models/tinykan_har.py` — depends on kan.py, classification_head.py, zsl_module.py

**Phase C: Baselines**
12. `models/baselines/knn.py` — no dependencies
13. `models/baselines/svm.py` — no dependencies
14. `models/baselines/random_forest.py` — no dependencies
15. `models/baselines/cnn1d.py` — no dependencies
16. `models/baselines/lstm.py` — no dependencies
17. `models/baselines/cnn_lstm.py` — no dependencies
18. `models/baselines/transformer.py` — no dependencies

**Phase D: Training Infrastructure**
19. `training/losses.py` — no dependencies
20. `evaluation/har_metrics.py` — no dependencies
21. `evaluation/zsl_metrics.py` — no dependencies
22. `training/train_kan.py` — depends on losses.py, har_metrics.py, zsl_metrics.py, tinykan_har.py

**Phase E: Evaluation**
23. `evaluation/statistical_tests.py` — no dependencies
24. `training/multi_seed.py` — depends on train_kan.py

**Phase F: Explainability**
25. `explainability/gradient_attribution.py` — no dependencies
26. `explainability/sensor_aggregation.py` — no dependencies
27. `explainability/temporal_aggregation.py` — no dependencies
28. `explainability/shap_approx.py` — no dependencies
29. `explainability/spline_viz.py` — no dependencies

**Phase G: Deployment**
30. `deployment/quantize.py` — no dependencies
31. `deployment/lut_generator.py` — no dependencies
32. `deployment/tflm_converter.py` — no dependencies
33. `deployment/mcu_profiler.py` — no dependencies

**Phase H: Experiments**
34. `experiments/exp1_har_performance.py` — depends on dataset.py, kan.py, baselines/*, train_kan.py, har_metrics.py
35. `experiments/exp2_zsl.py` — depends on dataset.py, tinykan_har.py, train_kan.py, zsl_metrics.py, zsl_module.py
36. `experiments/exp3_gamma_robustness.py` — depends on tinykan_har.py, zsl_metrics.py
37. `experiments/exp4_statistical.py` — depends on multi_seed.py, statistical_tests.py
38. `experiments/exp5_ablations.py` — depends on train_kan.py, ablations.py, quantize.py, lut_generator.py, mcu_profiler.py
39. `experiments/exp6_tinyml.py` — depends on quantize.py, lut_generator.py, tflm_converter.py, mcu_profiler.py
40. `experiments/exp7_viz.py` — depends on gradient_attribution.py, sensor_aggregation.py, temporal_aggregation.py, spline_viz.py
41. `experiments/exp8_depth_width.py` — depends on kan.py, train_kan.py

---

## Key Implementation Details per Module

### models/kan_layer.py — KANLayer

**Class:** `KANLayer(nn.Module)`

**Methods:**
- `__init__(d_in, d_out, k_splines=5, spline_order=3)` — Initialize W (d_out×d_in), b (d_out), theta (d_out×k_splines). Create B-spline basis.
- `_compute_basis(u)` — Given pre-activation u (scalar or tensor), evaluate B-spline basis functions B_k(u) for k=1..K
- `forward(x)` — Input x (batch, d_in). Compute u = Wx + b. For each output neuron j: compute phi_j(u_j) = sum_k theta[j,k] * B_k(u_j). Return (batch, d_out).
- `get_spline_function(j)` — Return the univariate spline phi_j as a callable over the pre-activation range
- `get_spline_derivative(j)` — Return derivative of phi_j w.r.t. u

**Key equations from paper:**
- Pre-activation: u^{(l)} = W^{(l)} x^{(l-1)} + b^{(l)} [Eq. 11]
- Spline output: x_j^{(l)} = phi_j^{(l)}(u_j^{(l)}) [Eq. 12]
- Spline parameterization: phi_j^{(l)}(u) = sum_k theta_{j,k}^{(l)} B_k^{(l)}(u) [Eq. 14]

### models/kan.py — KAN

**Class:** `KAN(nn.Module)`

**Methods:**
- `__init__(input_dim, layer_dims=[d0, d1, ..., dL], k_splines=5)` — Build L KANLayers with pyramidal dimensions
- `forward(x)` — Flatten input if 2D, pass through each KANLayer sequentially, return latent z = x^{(L)}
- `extract_splines()` — Return dict mapping (layer_idx, neuron_idx) -> (u_range, phi(u), dphi/du)

### models/zsl_module.py — ZSLModule

**Class:** `ZSLModule(nn.Module)`

**Methods:**
- `__init__(latent_dim, semantic_dim, n_seen, n_unseen, gamma=0.5)` — Initialize W_sem (semantic_dim × latent_dim), store semantic embeddings S_seen, S_unseen
- `project(z)` — h = W_sem @ z, normalized → (semantic_dim,)
- `compatibility(z, s_y)` — cos(h, s_y) = (h^T s_y) / (||h|| ||s_y||)
- `forward_zsl(z)` — Compute compatibility with all seen+unseen classes, apply calibration (subtract gamma from seen), return predicted class
- `set_semantic_embeddings(S_seen, S_unseen)` — Load semantic embeddings for seen and unseen classes

**Key equations from paper:**
- Projection: h = W_sem z [Eq. 28]
- Compatibility: g_phi(z, s_y) = cos(h, s_y) [Eq. 31]
- Calibration: g_tilde(z, s_y) = g(z, s_y) - gamma if y in Y_s else g(z, s_y) [Eq. 40]

### training/losses.py

**Functions:**
- `cross_entropy_loss(logits, labels)` — Standard softmax cross-entropy [Eq. 18]
- `alignment_loss(h_pred, h_true)` — L2 distance between projected and target semantic vectors [Eq. 30]
- `sem_ce_loss(compatibility_scores, labels, temperature)` — Semantic softmax cross-entropy [Eq. 35]
- `smoothness_loss(model)` — Sum of squared second differences of spline coefficients across all KAN layers [Eq. 21, 33]
- `total_loss(L_CE, L_align, L_sem_CE, model, lambda_ZSL, lambda_align, lambda_sem_CE, lambda_smooth)` — Combined loss [Eq. 19, 37, 25]

### deployment/quantize.py

**Functions:**
- `quantize_tensor(tensor, num_bits=8)` — Symmetric uniform quantization: delta = max(|tensor|) / (2^{b-1}), Q_b(w) = clip(round(w/delta), -2^{b-1}, 2^{b-1}), stored as int8
- `quantize_model(model, calib_data)` — Per-layer or per-tensor quantization with calibration data
- `fake_quantize_model(model)` — Add fake quantization nodes for quantization-aware training

### deployment/lut_generator.py

**Functions:**
- `generate_spline_lut(spline_fn, u_min, u_max, grid_size=256)` — Pre-sample phi(u) on uniform grid [u_min, u_max], quantize to int8, store as lookup table
- `generate_all_luts(model)` — Generate LUT for every spline function in every KAN layer
- `interpolate_lut(lut, u)` — Given input u, binary search LUT index, linearly interpolate between adjacent grid points

---

## Verification: No Circular Dependencies

The dependency graph is acyclic when ordered by the topological sort above. Key cycles avoided:
- `KANLayer` → `KAN` → `TinyKANHAR` → no cycle back to `KANLayer`
- `dataset.py` depends only on preprocessing primitives, not on models
- `train_kan.py` depends on losses, metrics (no circular import with models)
- Experiments depend on models, not vice versa
