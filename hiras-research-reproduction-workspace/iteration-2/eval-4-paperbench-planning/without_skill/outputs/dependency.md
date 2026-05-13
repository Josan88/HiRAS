# Dependency Model: TinyKAN-HAR

## Required Packages (requirements.txt format)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
pyyaml>=6.0
```

---

## Inter-File Dependencies (Topological Order)

### Tier 0 — Foundation (no dependencies)

1. **data/preprocessing.py**
   - Functions: `resample_to_target()`, `segment_windows()`, `separate_gravity()`, `normalize_channels()`, `interpolate_missing()`, `split_seen_unseen()`
   - Exports: `preprocess_dataset(dataset_name, raw_path) → HARDataset`
   - Depends on: NumPy, SciPy

2. **data/datasets.py**
   - Classes: `HARDataset` (torch.utils.data.Dataset)
   - Exports: `load_uci_har()`, `load_wisdm()`, `load_pamap2()`
   - Depends on: `preprocessing.py`, torch.utils.data

3. **models/kan_layer.py**
   - Classes: `KANLayer`
   - Methods: `forward(x)`, `compute_smoothness_reg()`
   - Depends on: torch

4. **models/classification_head.py**
   - Classes: `ClassificationHead`
   - Methods: `forward(z) → logits`
   - Depends on: torch

### Tier 1 — Core Model Components

5. **models/kan.py**
   - Classes: `KANFeatureExtractor`
   - Methods: `forward(x) → z`, `compute_reg()`
   - Depends on: `kan_layer.py`

6. **models/zsl_module.py**
   - Classes: `ZSLModule`
   - Methods: `forward(z)`, `compute_alignment_loss()`, `compute_semantic_softmax()`, `pure_zsl_inference()`, `gzsl_inference(gamma)`
   - Depends on: `models/kan.py` (receives z as input)

7. **models/explainability.py**
   - Classes: `ExplainabilityModule`
   - Methods: `compute_attributions()`, `aggregate_sensor()`, `aggregate_temporal()`, `compute_shap()`, `plot_splines()`
   - Depends on: `models/kan.py`, `models/kan_layer.py`

### Tier 2 — Full Model Assembly

8. **models/tinykan_har.py**
   - Classes: `TinyKANHAR`
   - Components: `KANFeatureExtractor`, `ClassificationHead`, `ZSLModule`, `ExplainabilityModule`
   - Methods: `forward_seen(x)`, `forward_zsl(x)`, `compute_loss()`, `explain(x)`
   - Depends on: `kan.py`, `classification_head.py`, `zsl_module.py`, `explainability.py`

9. **models/baselines/classical.py**
   - Classes: `KNNBaseline`, `SVMBaseline`, `RandomForestBaseline`
   - Methods: `fit()`, `predict()`
   - Depends on: scikit-learn, `data/preprocessing.py`

10. **models/baselines/cnn.py**
    - Classes: `CNN1DBaseline`
    - Methods: `forward(x) → logits`
    - Depends on: torch

11. **models/baselines/lstm.py**
    - Classes: `LSTMBaseline`, `CNNLSTMBaseline`
    - Methods: `forward(x) → logits`
    - Depends on: torch

12. **models/baselines/transformer.py**
    - Classes: `TransformerBaseline`
    - Methods: `forward(x) → logits`
    - Depends on: torch

### Tier 3 — Training

13. **training/loss.py**
    - Functions: `ce_loss()`, `zsl_alignment_loss()`, `semantic_softmax_loss()`, `combined_zsl_loss()`, `kan_regularization()`, `total_loss()`
    - Depends on: torch, `models/tinykan_har.py`

14. **training/optimizer.py**
    - Functions: `build_optimizer()`
    - Depends on: torch

15. **training/callbacks.py**
    - Classes: `EarlyStopping`, `MetricLogger`, `ModelCheckpoint`
    - Depends on: torch

16. **training/trainer.py**
    - Classes: `Trainer`
    - Methods: `train_seen()`, `train_zsl()`, `evaluate()`
    - Depends on: `loss.py`, `optimizer.py`, `callbacks.py`, `models/tinykan_har.py`

### Tier 4 — Quantization and TinyML

17. **quantization/quantize.py**
    - Functions: `quantize_model()`, `quantize_weights()`, `quantize_activations()`
    - Depends on: torch, torch.quantization

18. **quantization/qat.py**
    - Functions: `quantization_aware_training()`
    - Depends on: `quantize.py`

19. **quantization/lut.py**
    - Functions: `build_spline_luts()`, `evaluate_lut()`
    - Depends on: `models/kan_layer.py`, `quantize.py`

20. **tinyml/export.py**
    - Functions: `export_to_tflite()`, `generate_firmware()`
    - Depends on: `quantize.py`, `lut.py`

21. **tinyml/benchmark.py**
    - Functions: `benchmark_on_mcu()`
    - Depends on: `tinyml/export.py`

### Tier 5 — Evaluation

22. **evaluation/metrics.py**
    - Functions: `compute_har_metrics()`, `compute_zsl_metrics()`, `compute_gzsl_metrics()`, `compute_tinyml_metrics()`
    - Depends on: torch, scikit-learn, pandas

23. **evaluation/explainability_eval.py**
    - Functions: `generate_attribution_heatmap()`, `plot_sensor_relevance()`, `plot_temporal_relevance()`, `plot_spline_functions()`
    - Depends on: `evaluation/metrics.py`, `models/explainability.py`, matplotlib

24. **evaluation/ablations.py**
    - Functions: `run_ablation_variant()`, `run_all_ablations()`
    - Depends on: `models/tinykan_har.py`, `evaluation/metrics.py`, `training/trainer.py`

### Tier 6 — Scripts (Entry Points)

25. **scripts/train_seen.py**
    - Depends on: `training/trainer.py`, `models/tinykan_har.py`, `data/datasets.py`

26. **scripts/train_zsl.py**
    - Depends on: `training/trainer.py`, `models/tinykan_har.py`, `models/zsl_module.py`

27. **scripts/eval_seen.py**
    - Depends on: `evaluation/metrics.py`, `data/datasets.py`

28. **scripts/eval_zsl.py**
    - Depends on: `evaluation/metrics.py`, `models/zsl_module.py`

29. **scripts/eval_calibration.py**
    - Depends on: `evaluation/metrics.py`, `models/zsl_module.py`

30. **scripts/eval_multi_seed.py**
    - Depends on: `scripts/train_zsl.py`, `evaluation/metrics.py`

31. **scripts/visualize_explanations.py**
    - Depends on: `evaluation/explainability_eval.py`, `models/explainability.py`

32. **scripts/run_ablations.py**
    - Depends on: `evaluation/ablations.py`

33. **scripts/eval_semantic_ablation.py**
    - Depends on: `evaluation/ablations.py`, `models/zsl_module.py`

34. **scripts/eval_semantic_structure.py**
    - Depends on: `evaluation/ablations.py`, `models/zsl_module.py`

35. **scripts/deploy_tinyml.py**
    - Depends on: `tinyml/export.py`, `tinyml/benchmark.py`

36. **scripts/analyze_interpretations.py**
    - Depends on: `evaluation/explainability_eval.py`, `evaluation/metrics.py`

37. **main.py**
    - Depends on: all scripts above; orchestrates experiment selection and result logging

---

## Task Ordering (Topological Sort)

```
preprocessing.py → datasets.py
kan_layer.py → kan.py → classification_head.py
kan.py + zsl_module.py → tinykan_har.py
kan.py → explainability.py
datasets.py + tinykan_har.py → classical.py
datasets.py + tinykan_har.py → cnn.py / lstm.py / transformer.py
kan.py + classification_head.py + zsl_module.py → loss.py
loss.py → optimizer.py → callbacks.py → trainer.py
tinykan_har.py → quantize.py → qat.py
kan_layer.py + quantize.py → lut.py
quantize.py + lut.py → tinyml/export.py
tinyml/export.py → tinyml/benchmark.py
datasets.py + trainer.py → eval_seen.py / eval_zsl.py
trainer.py + tinykan_har.py → eval_calibration.py
trainer.py + tinykan_har.py → eval_multi_seed.py
trainer.py + tinykan_har.py → run_ablations.py
trainer.py + tinykan_har.py + zsl_module.py → eval_semantic_ablation.py
trainer.py + tinykan_har.py + zsl_module.py → eval_semantic_structure.py
trainer.py + tinykan_har.py → visualize_explanations.py
trainer.py + tinykan_har.py → analyze_interpretations.py
export.py + benchmark.py → deploy_tinyml.py
(all scripts) → main.py
```

**Note:** `explainability.py` and `classical.py`/`cnn.py`/`lstm.py`/`transformer.py` are independent of each other and can be implemented in parallel once `kan.py` and `datasets.py` exist. No circular dependencies exist in this architecture.

---

## Key Classes and Functions Summary

| File | Class/Function | Description |
|------|---------------|-------------|
| `kan_layer.py` | `KANLayer` | Single KAN layer with linear mixing + spline bank |
| `kan_layer.py` | `KANLayer.forward()` | Eq. 11–14: linear mix then univariate splines |
| `kan_layer.py` | `KANLayer.smoothness_reg()` | Eq. 21: discrete second-difference penalty |
| `kan.py` | `KANFeatureExtractor` | L-layer KAN producing latent z ∈ ℝ^{d_L} |
| `zsl_module.py` | `ZSLModule` | Semantic projection + alignment + semantic softmax |
| `zsl_module.py` | `ZSLModule.compatibility()` | Eq. 31: cosine similarity g_φ(z, s_y) |
| `zsl_module.py` | `ZSLModule.gzsl_inference()` | Eq. 39–40: calibrated gZSL prediction |
| `tinykan_har.py` | `TinyKANHAR` | Full model integrating all submodules |
| `tinykan_har.py` | `TinyKANHAR.total_loss()` | Eq. 37: L_CE + λ_ZSL*L_ZSL + R_KAN |
| `quantize.py` | `quantize_to_int8()` | Eq. 42: symmetric uniform quantization |
| `lut.py` | `build_spline_luts()` | Pre-sample splines, embed in LUT arrays |
| `preprocessing.py` | `preprocess_dataset()` | Full pipeline: resample → segment → normalize |
| `metrics.py` | `compute_har_metrics()` | Accuracy, precision, recall, F1 (macro/micro) |
| `metrics.py` | `compute_gzsl_metrics()` | Acc_seen, Acc_unseen, harmonic mean H |
| `metrics.py` | `compute_tinyml_metrics()` | Model size, RAM, latency, energy |
| `loss.py` | `combined_zsl_loss()` | Eq. 37: full training objective |
| `loss.py` | `kan_regularization()` | Eq. 20 (weight decay) + Eq. 21 (smoothness) |
| `explainability.py` | `ExplainabilityModule.attributions()` | Gradient × input attributions |