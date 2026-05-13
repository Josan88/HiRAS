# Phase 3: Dependency Modelling - TinyKAN-HAR Reproduction

## Required Packages

```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
pyyaml>=6.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
tqdm>=4.65.0
```

## Inter-File Dependencies

```
config.py
├── config.yaml (data)

datasets/base.py
├── datasets/uci_har.py
├── datasets/wisdm.py
└── datasets/pamap2.py

data/preprocessor.py
├── datasets/base.py

models/kan_layer.py
├── (no internal dependencies - core primitive)

models/kan.py
├── models/kan_layer.py

models/classifier.py
├── (no internal dependencies)

models/zsl_module.py
├── models/kan.py
└── data/semantic_embeddings.py

models/explainer.py
├── models/kan.py

training/trainer.py
├── models/kan.py
├── models/classifier.py
├── models/zsl_module.py
├── training/losses.py
└── training/optimizer.py

evaluation/evaluator.py
├── models/kan.py
├── models/zsl_module.py
├── evaluation/metrics.py
└── training/trainer.py (for loading checkpoints)

experiments/exp1_seen_classes.py
├── datasets/*
├── models/kan.py
├── models/classifier.py
├── models/baselines/*
├── training/trainer.py
└── evaluation/evaluator.py

experiments/exp2_zsl.py
├── datasets/*
├── models/kan.py
├── models/zsl_module.py
├── training/trainer.py
├── data/semantic_embeddings.py
└── evaluation/evaluator.py

experiments/exp5_ablation.py
├── utils/quantization.py
├── utils/pruning.py
└── utils/lut.py
```

---

## Classes, Methods, and Functions to Implement

### datasets/base.py
```python
class BaseDataset(ABC):
    @abstractmethod
    def load(self) -> Tuple[Dataset, Dataset, Dataset]: ...
    
    @abstractmethod
    def get_split(self, split: str) -> Dataset: ...
    
    def preprocess(self, windows: np.ndarray) -> np.ndarray: ...
    
    def get_seen_unseen_split(self) -> Tuple[List[str], List[str]]: ...
```

### datasets/uci_har.py
```python
class UCI_HAR_Dataset(BaseDataset):
    # Inherits BaseDataset
    # Specifics: 30 subjects, 6 activities, 50Hz
    # Split: 21 train / 4 val / 9 test
    # Seen: {WALKING, SITTING, STANDING, LAYING}
    # Unseen: {WALKING_UPSTAIRS, WALKING_DOWNSTAIRS}
```

### datasets/wisdm.py
```python
class WISDM_Dataset(BaseDataset):
    # 51 subjects, 18 activities, 20Hz
    # Split: 35 train / 8 val / 8 test
    # Seen: 14 activities
    # Unseen: {jogging, walking_upstairs, walking_downstairs, jumping}
```

### datasets/pamap2.py
```python
class PAMAP2_Dataset(BaseDataset):
    # 9 subjects, 18 activities, 100Hz→50Hz
    # Split: 6 train / 1 val / 2 test
    # Seen: 14 activities
    # Unseen: {running, rope_jumping, vacuum_cleaning, ironing}
    # Handles heart-rate sensor interpolation
```

### data/preprocessor.py
```python
def resample_to_target(signals: np.ndarray, original_fs: int, target_fs: int = 50) -> np.ndarray: ...
def separate_gravity(acc: np.ndarray, fs: int = 50) -> Tuple[np.ndarray, np.ndarray]: ...
def normalize_per_channel(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray: ...
def segment_windows(signals: np.ndarray, labels: np.ndarray, T: int, stride: int) -> Tuple[np.ndarray, np.ndarray]: ...
def interpolate_missing(r: np.ndarray, m: np.ndarray) -> np.ndarray: ...
```

### models/kan_layer.py
```python
class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_knots: int = 5, k: int = 3): ...
    def initialize_spline_identity(self): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # u = W @ x + b
        # x_out = sum_k theta_{j,k} * B_k(u_j) for each neuron j
```

### models/kan.py
```python
class KANFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, layer_dims: List[int], num_knots: int = 5, k: int = 3, dropout: float = 0.1): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequentially apply L KAN layers
        # Return final latent z ∈ R^{d_L}
```

### models/classifier.py
```python
class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int): ...
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # s = V @ z + c (logits)
        # return softmax(s) or raw logits
```

### models/zsl_module.py
```python
class ZSLModule(nn.Module):
    def __init__(self, latent_dim: int, semantic_dim: int, num_seen_classes: int): ...
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h = W_sem @ z
        # g(y) = cos(h, s_y) for each class y
        # Return compatibility scores for all classes
    def compute_zsl_loss(self, h: torch.Tensor, y_true: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # L_align + L_sem_CE
    def calibrate_scores(self, g: torch.Tensor, y_seen_mask: torch.Tensor, gamma: float) -> torch.Tensor:
        # g_calibrated = g - gamma for seen classes
```

### data/semantic_embeddings.py
```python
def get_attribute_embeddings(activity_names: List[str]) -> np.ndarray: ...
def get_text_embeddings(activity_names: List[str], model_name: str = "TODO") -> np.ndarray: ...
def get_hybrid_embeddings(activity_names: List[str]) -> np.ndarray: ...
def create_semantic_matrix(activity_names: List[str], unseen_names: List[str]) -> np.ndarray: ...
```

### training/losses.py
```python
def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
def zsl_alignment_loss(h: torch.Tensor, s_y: torch.Tensor) -> torch.Tensor: ...
def semantic_softmax_loss(a: torch.Tensor, y_true: torch.Tensor, temperature: float = 1.0) -> torch.Tensor: ...
def combined_loss(L_CE: torch.Tensor, L_ZSL: torch.Tensor, lambda_zsl: float) -> torch.Tensor: ...
```

### training/optimizer.py
```python
def get_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Adam: ...
def get_scheduler(optimizer, schedule_type: str = "cosine", total_epochs: int = 100) -> Any: ...
```

### evaluation/metrics.py
```python
def accuracy(preds: np.ndarray, labels: np.ndarray) -> float: ...
def precision_recall_f1(preds: np.ndarray, labels: np.ndarray, average: str = "macro") -> Tuple[float, float, float]: ...
def zsl_accuracy(compatibility_scores: np.ndarray, true_labels: np.ndarray, unseen_mask: np.ndarray) -> float: ...
def gzsl_harmonic_mean(acc_seen: float, acc_unseen: float) -> float: ...
```

### evaluation/evaluator.py
```python
class Evaluator:
    def __init__(self, model, device: str = "cuda"): ...
    def evaluate_seen_classes(self, dataset) -> Dict[str, float]: ...
    def evaluate_zsl(self, dataset, semantic_matrix) -> Dict[str, float]: ...
    def evaluate_gzsl(self, dataset, semantic_matrix, gamma: float) -> Dict[str, float]: ...
```

### utils/quantization.py
```python
def quantize_tensor(x: torch.Tensor, num_bits: int = 8) -> Tuple[torch.Tensor, float]: ...
def quantize_model(model: nn.Module, num_bits: int = 8) -> nn.Module: ...
def fake_quantize_forward(x: torch.Tensor, scale: float, num_bits: int = 8) -> torch.Tensor: ...
```

### utils/lut.py
```python
def generate_spline_lut(spline_func, input_range: Tuple[float, float], num_points: int = 256) -> np.ndarray: ...
def generate_model_luts(model: KANFeatureExtractor, num_points: int = 256) -> Dict[str, np.ndarray]: ...
def lut_forward(lut: np.ndarray, input_value: float, scale: float) -> float: ...
```

### utils/pruning.py
```python
def structured_prune_layer(layer: nn.Linear, prune_rate: float) -> nn.Module: ...
def structured_prune_kan(kan: KANFeatureExtractor, prune_rate: float) -> KANFeatureExtractor: ...
```

---

## Task Ordering (Topological Sort)

**Phase A: Foundation (implement first, no dependencies)**
1. `config.yaml` - all hyperparameters
2. `datasets/base.py` - abstract interface
3. `datasets/uci_har.py` - first dataset
4. `datasets/wisdm.py` - second dataset
5. `datasets/pamap2.py` - third dataset
6. `data/preprocessor.py` - shared preprocessing
7. `data/semantic_embeddings.py` - ZSL semantic module data prep

**Phase B: Core Models (depends on Phase A)**
8. `models/kan_layer.py` - KAN primitive
9. `models/kan.py` - KAN stack (depends on 8)
10. `models/classifier.py` - classification head
11. `models/zsl_module.py` - ZSL module (depends on 9)
12. `models/explainer.py` - explainability (depends on 9)

**Phase C: Baselines (depends on Phase A, B partially)**
13. `models/baselines/knn.py`
14. `models/baselines/svm.py`
15. `models/baselines/random_forest.py`
16. `models/baselines/cnn1d.py`
17. `models/baselines/lstm.py`
18. `models/baselines/cnn_lstm.py`
19. `models/baselines/transformer.py`

**Phase D: Training Infrastructure**
20. `training/losses.py`
21. `training/optimizer.py`
22. `training/early_stopping.py`
23. `training/trainer.py`

**Phase E: Evaluation**
24. `evaluation/metrics.py`
25. `evaluation/evaluator.py`
26. `evaluation/tinyml_metrics.py`

**Phase F: Utilities**
27. `utils/quantization.py`
28. `utils/pruning.py`
29. `utils/lut.py`
30. `utils/export.py`

**Phase G: Experiments**
31. `experiments/exp1_seen_classes.py`
32. `experiments/exp2_zsl.py`
33. `experiments/exp3_gamma.py`
34. `experiments/exp4_significance.py`
35. `experiments/exp5_ablation.py`
36. `experiments/exp6_tinyml.py`

---

## Import Graph (Summary)

```
config.py ──────────────────────────────────────────────────┐
                                                           │
datasets/base.py ───┬──→ datasets/uci_har.py               │
                    ├──→ datasets/wisdm.py                  │
                    └──→ datasets/pamap2.py ───────────────┤
                                                           │
data/preprocessor.py ─────────────────────────────────────┤
data/semantic_embeddings.py ────────────────────────────────┤
                                                           │
models/kan_layer.py ───→ models/kan.py ────────────────────┤
                                   │                        │
                                   ├──→ models/classifier.py│
                                   ├──→ models/zsl_module.py
                                   └──→ models/explainer.py
                                                           │
models/baselines/* ────────────────────────────────────────┤
                                                           │
training/losses.py ───→ training/trainer.py ───────────────┤
training/optimizer.py ─→ training/trainer.py ──────────────┤
                                                           │
evaluation/metrics.py ───→ evaluation/evaluator.py ────────┤
                                                           │
utils/quantization.py ───┐                                │
utils/pruning.py ────────┼──→ experiments/* ──────────────┤
utils/lut.py ────────────┤                                │
utils/export.py ─────────┘                                │
```

No circular imports exist. All dependencies flow from data/preprocessing up through models, then training, then evaluation, then experiments.